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
from .metadata import _object_id_to_name, _object_name_to_id, _predicate_id_to_name, _predicate_name_to_id
from ...utils import apples_to_apples


class HicoDet(torch.utils.data.Dataset):
    objects: Tuple[str] = tuple(_object_id_to_name)
    predicates: Tuple[str] = tuple(_predicate_id_to_name)

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

        self.graphs = None
        self.transforms = transforms

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        if self.graphs is not None:
            graph = self.graphs[item]
        else:
            graph = self.make_graph(torch.load(self.paths[item]))

        if self.transforms is not None:
            graph = self.transforms(graph)

        if graph.num_nodes == 0:
            # If a graph has 0 nodes and it's put last in the the batch formed by
            # Batch.from_data_list(...) it will cause a miscount in batch.num_graphs
            logger.warning(f'Loaded graph without nodes for: {graph.filename}')

        return graph

    def make_graph(self, sample: HicoDetSample) -> Data:
        if self.input_mode is HicoDet.InputMode.GT:
            instances = sample.gt_instances
            visual_relations = sample.gt_visual_relations_full
        elif self.input_mode is HicoDet.InputMode.D2:
            instances = sample.d2_instances
            visual_relations = sample.d2_visual_relations_full
        else:
            raise ValueError(f'Unknown input mode: {self.input_mode}')

        unique_predicates = torch.unique(sample.gt_visual_relations.predicate_classes, sorted=False)
        target_bce = torch.zeros(len(HicoDet.predicates), dtype=torch.float)
        target_bce[unique_predicates] = 1.
        target_rank = torch.full((len(HicoDet.predicates),), fill_value=-1, dtype=torch.long)
        target_rank[:len(unique_predicates)] = unique_predicates

        graph = Data(
            num_nodes=len(instances),
            node_conv_features=instances.conv_features,
            node_linear_features=instances.linear_features,
            edge_index=visual_relations.relation_indexes,
            edge_attr=visual_relations.linear_features,
            target_bce=target_bce.unsqueeze_(dim=0),
            target_rank=target_rank.unsqueeze_(dim=0),
        )

        return graph

    def load_eager(self):
        if self.graphs is not None:
            return

        vr_count = 0
        inst_count = 0
        self.graphs = []
        start_time = time.perf_counter()
        for p in tqdm(self.paths, desc='Loading', unit='g'):
            g = self.make_graph(torch.load(p))
            vr_count += g.target_bce.sum().int().item()
            inst_count += g.num_nodes
            self.graphs.append(g)
        logger.info(f'Loaded '
                    f'{inst_count:,} {self.input_mode.name.lower()} instances with '
                    f'{vr_count:,} gt visual relations '
                    f'from {len(self.paths):,} .pth files '
                    f'in {time.perf_counter() - start_time:.1f}s')

    @classmethod
    @apples_to_apples
    def object_id_to_name(cls, ids: Iterable[int]):
        return [_object_id_to_name[i] for i in ids]

    @classmethod
    @apples_to_apples
    def object_name_to_id(cls, names: Iterable[str]):
        return [_object_name_to_id[n] for n in names]

    @classmethod
    @apples_to_apples
    def predicate_id_to_name(cls, ids: Iterable[int]):
        return [_predicate_id_to_name[i] for i in ids]

    @classmethod
    @apples_to_apples
    def predicate_name_to_id(cls, names: Iterable[str]):
        return [_predicate_name_to_id[n] for n in names]