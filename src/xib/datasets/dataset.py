from __future__ import annotations

import random
from enum import Enum
from typing import Tuple, Union, Optional, Callable, Sequence, Any, Dict

import torch.utils
import torch.utils.data
from PIL import Image
from detectron2.data.catalog import Metadata
from loguru import logger
from torch_geometric.data import Data
from torchvision.transforms import (
    Resize,
    Compose,
    Normalize,
    ToTensor,
    RandomResizedCrop,
)

from xib.structures.instances import to_data_dict as instance_to_data_dict
from xib.utils import NamedEnumMixin
from .folder import DatasetFolder
from .sample import VrSample


class VrDataset(torch.utils.data.Dataset):
    """
    Loads VrSample from a dataset folder and produces torch_geometric.data.Data objects.

    When creating the input graphs, it is possible to choose between
    ground-truth boxes and their fully-connected graph or
    detectron2 boxes and their fully-connected graph.

    Also, it supports loading all the samples eagerly before training.
    """

    input_mode: InputMode

    class InputMode(NamedEnumMixin, Enum):
        GT = 0
        D2 = 1

    def __init__(
        self,
        folder: DatasetFolder,
        input_mode: Union[str, InputMode],
        metadata: Metadata,
        transforms: Optional[Sequence[Callable]] = None,
    ):
        self.input_mode = VrDataset.InputMode.get(input_mode)
        self.folder = folder
        self.metadata = metadata
        self.transforms = transforms if transforms is not None else []
        self._blacklist = set()

    def __len__(self):
        return len(self.folder)

    def __getitem__(self, item) -> Tuple[Data, Data, str]:
        if item in self._blacklist:
            return self[random.randrange(0, len(self))]

        sample = self.folder[item]

        input_graph, target_graph = self.make_graphs(sample)

        for t in self.transforms:
            input_graph, target_graph = t(input_graph, target_graph, sample.filename)

        if input_graph.n_nodes == 0:
            # If a graph has 0 nodes and it's put last in the the batch formed by
            # Batch.from_data_list(...) it will cause a miscount in batch.num_graphs
            logger.warning(f"Blacklisting graph without nodes: {sample.filename}")
            self._blacklist.add(item)
            return self[random.randrange(0, len(self))]

        if input_graph.n_edges == 0:
            if self.metadata.name.endswith("train"):
                # If a graph has 0 edges it's useless for training
                logger.warning(f"Blacklisting graph without edges: {sample.filename}")
                self._blacklist.add(item)
                return self[random.randrange(0, len(self))]
            else:
                logger.warning(f"Found a graph without edges: {sample.filename}")

        # torch_geometric does not support string attributes when collating
        return input_graph, target_graph, sample.filename

    def load_eager(self):
        self.folder.load_eager()

    def make_graphs(self, sample: VrSample) -> Tuple[Data, Data]:
        # region Input: fully connected graph with visual features
        if self.input_mode is VrDataset.InputMode.GT:
            input_instances = sample.gt_instances
            input_relations = sample.gt_visual_relations_full
        elif self.input_mode is VrDataset.InputMode.D2:
            input_instances = sample.d2_instances
            input_relations = sample.d2_visual_relations_full
        else:
            raise ValueError(f"Unknown input mode: {self.input_mode}")

        # TODO remove this after reding preprocessing of VRD
        if not input_instances.has("scores"):
            input_instances.scores = input_instances.probabs.gather(
                dim=1, index=input_instances.classes[:, None]
            ).squeeze(dim=1)

        input_graph = Data(
            # When collating this will be summed to num_nodes of other graphs
            num_nodes=len(input_instances),
            # Instead, these two will be concatenated
            n_nodes=len(input_instances),
            n_edges=len(input_relations),
            **instance_to_data_dict(input_instances, prefix="object"),
            **input_relations.to_data_dict(),
        )
        input_graph.object_linear_features = torch.cat(
            (input_graph.object_linear_features, input_graph.object_probabs), dim=1
        )
        # endregion

        # region Target: ground-truth relations
        target_graph = Data(
            # When collating this will be summed to num_nodes of other graphs
            num_nodes=len(sample.gt_instances),
            # Instead, these two will be concatenated
            n_nodes=len(sample.gt_instances),
            n_edges=len(sample.gt_visual_relations),
            # Reshape these two so they are stacked on the first dimension
            **instance_to_data_dict(sample.gt_instances, prefix="object"),
            **sample.gt_visual_relations.to_data_dict(),
        )
        # endregion

        # Debug util
        """
        print(sample.filename)
        for n, g in {"input_graph": input_graph, "target_graph": target_graph}.items():
            print(n)
            for key, value in g.__dict__.items():
                if value is not None:
                    if torch.is_tensor(value):
                        v = str(tuple(value.shape))
                        t = str(value.dtype).replace("torch.", "t.")
                    else:
                        v = str(value)
                        t = type(value).__name__
                    print(f"- {key:<25} {v:>18} {t:>10}")
            print()
        """

        return input_graph, target_graph


class PcDataset(torch.utils.data.Dataset):
    def __init__(
        self, data_dicts: Sequence[Dict[str, Any]], metadata: Metadata, transforms=None
    ):
        self.data_dicts = data_dicts
        self.metadata = metadata
        self.num_classes = len(metadata.predicate_classes)
        self.transforms = Compose(
            [
                *(transforms if transforms is not None else []),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.data_dicts)

    def __getitem__(self, item):
        d = self.data_dicts[item]

        img = Image.open(d["file_name"]).convert("RGB")
        img = self.transforms(img)

        unique_predicates = torch.tensor(
            list(set(r["category_id"] for r in d["relations"])), dtype=torch.long
        )
        target_bce = torch.zeros(self.num_classes, dtype=torch.float).scatter_(
            dim=0, index=unique_predicates, value=1.0
        )
        target_rank = torch.constant_pad_nd(
            unique_predicates,
            pad=(0, self.num_classes - len(unique_predicates)),
            value=-1,
        )

        return img, {"bce": target_bce, "rank": target_rank}, d["file_name"]
