from __future__ import annotations

import random
from enum import Enum
from typing import Tuple, Union, Optional, Callable

import torch.utils
import torch.utils.data
from loguru import logger
from torch_geometric.data import Data

from .sample import VrSample
from .folder import DatasetFolder
from xib.structures.instances import to_data_dict as instance_to_data_dict
from xib.utils import NamedEnumMixin


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
        transforms: Optional[Callable] = None,
    ):
        self.input_mode = VrDataset.InputMode.get(input_mode)
        self.folder = folder
        self.transforms = transforms
        self._blacklist = set()

    def __len__(self):
        return len(self.folder)

    def __getitem__(self, item) -> Tuple[Data, Data, str]:
        if item in self._blacklist:
            return self[random.randrange(0, len(self))]

        sample = self.folder[item]

        input_graph, target_graph = self.make_graphs(sample)

        if self.transforms is not None:
            input_graph, target_graph = self.transforms(input_graph, target_graph)

        if input_graph.num_nodes == 0:
            # If a graph has 0 nodes and it's put last in the the batch formed by
            # Batch.from_data_list(...) it will cause a miscount in batch.num_graphs
            logger.warning(f"Blacklisting graph without nodes for: {sample.filename}")
            self._blacklist.add(item)
            return self[random.randrange(0, len(self))]

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

        input_graph = Data(
            # When collating this will be summed to num_nodes of other graphs
            num_nodes=len(input_instances),
            # Instead, these two will be concatenated
            n_nodes=len(input_instances),
            n_edges=len(input_relations),
            **instance_to_data_dict(input_instances, prefix="object"),
            **input_relations.to_data_dict(),
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
