from __future__ import annotations

import time

from enum import Enum
from pathlib import Path
from typing import Tuple, Union, Optional, Callable, Iterable

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
        elif not isinstance(input_mode, HicoDet.InputMode):
            raise ValueError(f'Invalid input mode: {self.input_mode}')

        folder = Path(folder).expanduser().resolve()
        self.paths = sorted(p for p in folder.iterdir() if p.suffix == '.pth')
        if len(self.paths) == 0:
            logger.warning(f'Empty dataloader: no .pth file found in {folder}')

        self.samples = []
        self.loaded = False
        self.transforms = transforms

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        if self.loaded:
            graph, filename = self.samples[item]
        else:
            sample = torch.load(self.paths[item])
            graph = self.make_graph(sample)
            filename = sample.filename

        if self.transforms is not None:
            graph = self.transforms(graph)

        if graph.num_nodes == 0:
            # If a graph has 0 nodes and it's put last in the the batch formed by
            # Batch.from_data_list(...) it will cause a miscount in batch.num_graphs
            logger.warning(f'Loaded graph without nodes for: {graph.filename}')

        # string attributes require separate handling when collating
        return graph, filename

    def load_eager(self):
        if self.loaded:
            return

        gt_vr_count = 0
        gt_inst_count = 0
        start_time = time.perf_counter()
        for path in tqdm(self.paths, desc='Loading', unit='g'):
            sample = self.load_sample(path)
            graph = self.make_graph(sample)
            gt_vr_count += graph.num_target_edges
            gt_inst_count += graph.num_target_nodes
            self.samples.append((graph, sample.filename))

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

    def make_graph(self, sample: HicoDetSample) -> Data:
        # region Input: fully connected graph with visual features
        if self.input_mode is HicoDet.InputMode.GT:
            input_instances = sample.gt_instances
            input_relations = sample.gt_visual_relations_full
        elif self.input_mode is HicoDet.InputMode.D2:
            input_instances = sample.d2_instances
            input_relations = sample.d2_visual_relations_full
        else:
            raise ValueError(f'Unknown input mode: {self.input_mode}')

        input_graph = {
            'num_input_nodes': len(input_instances),
            'num_input_edges': len(input_relations),
            **instance_to_data_dict(input_instances, prefix='input_object'),
            **input_relations.to_data_dict(prefix='input')
        }
        # endregion

        # region Target: ground-truth relations + encoding of unique predicates
        unique_predicates = torch.unique(sample.gt_visual_relations.predicate_classes, sorted=False)
        target_bce = torch.zeros(len(PREDICATES), dtype=torch.float).scatter_(dim=0, index=unique_predicates, value=1.)
        target_rank = torch.full((len(PREDICATES),), fill_value=-1, dtype=torch.long)
        target_rank[:len(unique_predicates)] = unique_predicates

        target_graph = {
            'num_target_nodes': len(sample.gt_instances),
            'num_target_edges': len(sample.gt_visual_relations),
            'target_predicate_bce': target_bce[None, :],
            'target_predicate_rank': target_rank[None, :],
            **instance_to_data_dict(sample.gt_instances, prefix='target_object'),
            **sample.gt_visual_relations.to_data_dict(prefix='target'),
        }
        # endregion

        graph = Data(
            # Otherwise collate will raise error
            num_nodes=input_graph['num_input_nodes'],
            **input_graph,
            **target_graph,
        )

        return graph
