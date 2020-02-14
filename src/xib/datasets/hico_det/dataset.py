from __future__ import annotations

import time

from enum import Enum
from pathlib import Path
from typing import Tuple, Union, Optional, Callable, Iterable, List

import torch
import torch.utils
import torch.utils.data

from tqdm import tqdm
from loguru import logger
from torch_geometric.data import Data

from .sample import HicoDetSample
from .metadata import OBJECTS, PREDICATES
from ...structures.instances import to_data_dict as instance_to_data_dict


class HicoDet(torch.utils.data.Dataset):
    input_mode: InputMode

    class InputMode(Enum):
        GT = 0
        D2 = 1

    def __init__(
            self,
            folder: Union[str, Path],
            input_mode: Union[str, InputMode],
            transforms: Optional[Callable] = None
    ):
        if isinstance(input_mode, str):
            self.input_mode = HicoDet.InputMode[input_mode]
        else:
            if not isinstance(input_mode, HicoDet.InputMode):
                raise ValueError(f'Invalid input mode: {self.input_mode}')
            self.input_mode = input_mode

        folder = Path(folder).expanduser().resolve()
        self.paths = sorted(p for p in folder.iterdir() if p.suffix == '.pth')
        if len(self.paths) == 0:
            logger.warning(f'Empty dataloader: no .pth file found in {folder}')

        self.samples: List[Tuple[Data, Data, str]] = []
        self.loaded = False
        self.transforms = transforms

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item) -> Tuple[Data, Data, str]:
        if self.loaded:
            input_graph, target_graph, filename = self.samples[item]
        else:
            sample = torch.load(self.paths[item])
            input_graph, target_graph = self.make_graphs(sample)
            filename = sample.filename

        if self.transforms is not None:
            input_graph = self.transforms(input_graph)

        if input_graph.num_nodes == 0:
            # If a graph has 0 nodes and it's put last in the the batch formed by
            # Batch.from_data_list(...) it will cause a miscount in batch.num_graphs
            logger.warning(f'Loaded graph without nodes for: {filename}')

        # torch_geometric does not support string attributes when collating
        return input_graph, target_graph, filename

    def load_eager(self):
        if self.loaded:
            return

        gt_vr_count = 0
        gt_inst_count = 0
        start_time = time.perf_counter()

        for path in tqdm(self.paths, desc='Loading', unit='g'):
            sample = self.load_sample(path)
            input_graph, target_graph = self.make_graphs(sample)
            gt_vr_count += target_graph.n_edges
            gt_inst_count += target_graph.n_edges
            self.samples.append((input_graph, target_graph, sample.filename))

        self.loaded = True
        logger.info(
            f'Loaded '
            f'{gt_inst_count:,} {self.input_mode.name.lower()} instances with '
            f'{gt_vr_count:,} gt visual relations '
            f'from {len(self.paths):,} .pth files '
            f'in {time.perf_counter() - start_time:.1f}s'
        )

    @staticmethod
    def load_sample(path: Path) -> HicoDetSample:
        sample: HicoDetSample = torch.load(path)
        sample.gt_visual_relations.predicate_vocabulary = PREDICATES
        sample.gt_visual_relations.object_vocabulary = OBJECTS

        return sample

    def make_graphs(self, sample: HicoDetSample) -> Tuple[Data, Data]:
        # region Input: fully connected graph with visual features
        if self.input_mode is HicoDet.InputMode.GT:
            input_instances = sample.gt_instances
            input_relations = sample.gt_visual_relations_full
        elif self.input_mode is HicoDet.InputMode.D2:
            input_instances = sample.d2_instances
            input_relations = sample.d2_visual_relations_full
        else:
            raise ValueError(f'Unknown input mode: {self.input_mode}')

        input_graph = Data(
            # When collating this will be summed to num_nodes of other graphs
            num_nodes=len(input_instances),
            # Instead, these two will be concatenated
            n_nodes=len(input_instances),
            n_edges=len(input_relations),
            **instance_to_data_dict(input_instances, prefix='object'),
            **input_relations.to_data_dict(),
        )
        # endregion

        # region Target: ground-truth relations + encoding of unique predicates
        unique_preds = torch.unique(sample.gt_visual_relations.predicate_classes, sorted=False)
        target_bce = torch.zeros(len(PREDICATES), dtype=torch.float).scatter_(dim=0, index=unique_preds, value=1.)
        target_rank = torch.constant_pad_nd(unique_preds, pad=(0, len(PREDICATES) - len(unique_preds)), value=-1)

        target_graph = Data(
            # When collating this will be summed to num_nodes of other graphs
            num_nodes=len(sample.gt_instances),
            # Instead, these two will be concatenated
            n_nodes=len(sample.gt_instances),
            n_edges=len(sample.gt_visual_relations),
            # Reshape these two so they are stacked on the first dimension
            predicate_bce=target_bce[None, :],
            predicate_rank=target_rank[None, :],
            **instance_to_data_dict(sample.gt_instances, prefix='object'),
            **sample.gt_visual_relations.to_data_dict(),
        )
        # endregion

        # Debug util
        """
        print(sample.filename)
        for n, g in {'input_graph': input_graph, 'target_graph': target_graph}.items():
            print(n)
            for key, value in g.__dict__.items():
                if value is not None:
                    if torch.is_tensor(value):
                        v = str(tuple(value.shape))
                        t = str(value.dtype).replace('torch.', 't.')
                    else:
                        v = str(value)
                        t = type(value).__name__
                    print(f'- {key:<25} {v:>18} {t:>10}')
            print()
        """

        return input_graph, target_graph
