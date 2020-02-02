import time
import dataclasses
from pathlib import Path
from typing import Union, Iterable, List, Tuple, Dict, Optional, Callable

import torch
from torch_geometric.data import Data

from loguru import logger
from detectron2.structures import Instances

from ..utils import apples_to_apples
from ..structures import VisualRelations


@dataclasses.dataclass
class HicoDetSample(object):
    # Metadata
    filename: Union[Path, str]
    img_size: Tuple[int, int, int]

    # Ground-truth visual relations
    gt_visual_relations: Optional[VisualRelations] = None

    # Input features: image features, detection boxes, box features
    feature_pyramid: Optional[Dict[str, torch.Tensor]] = None
    detections: Optional[Instances] = None

    # Precomputed graph features
    node_conv_features: Optional[torch.Tensor] = None
    node_linear_features: Optional[torch.Tensor] = None
    edge_linear_features: Optional[torch.Tensor] = None
    edge_indices: Optional[torch.Tensor] = None


class HicoDet(torch.utils.data.Dataset):
    objects: Tuple[str]
    predicates: Tuple[str]

    def __init__(self, folder: Union[str, Path], transforms: Optional[Callable] = None):
        folder = Path(folder).expanduser().resolve()
        self.paths = [p for p in folder.iterdir() if p.suffix == '.pth']
        self.samples = None
        self.transforms = transforms

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        if self.samples is not None:
            sample = self.samples[item]
        else:
            sample = torch.load(self.paths[item])

        target = self._visual_relations_to_binary_targets(sample.gt_visual_relations)

        graph = Data(
            num_nodes=len(sample.detections),
            node_conv_features=sample.node_conv_features,
            node_linear_features=sample.node_linear_features,
            edge_index=sample.edge_indices,
            edge_attr=sample.edge_linear_features,
            target=target,
        )

        if self.transforms is not None:
            graph = self.transforms(graph)

        return graph

    def load_eager(self):
        start_time = time.perf_counter()
        self.samples = [torch.load(p) for p in self.paths]
        logger.info(f'Loaded {sum(len(s.gt_visual_relations) for s in self.samples):,} visual relations from '
                    f'{len(self.samples):,} images in {time.perf_counter() - start_time:.1f}s')


    @staticmethod
    def _visual_relations_to_binary_targets(visual_relations: VisualRelations) -> torch.Tensor:
        target = torch.zeros(len(HicoDet.predicates), dtype=torch.float)
        target[visual_relations.predicate_classes] = 1.
        return target.unsqueeze(dim=0)

    @classmethod
    def from_pytorch(cls, folder: Union[str, Path]):
        start_time = time.perf_counter()
        folder = Path(folder).expanduser().resolve()
        samples = [torch.load(f) for f in folder.iterdir() if f.suffix == '.pt']
        logger.info(f'Loaded {sum(len(s["visual_relations"]) for s in samples):,} visual relations from '
                    f'{len(samples):,} images in {time.perf_counter() - start_time:.1f}s')
        return cls(samples)

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


_object_id_to_name = (
    'person',
    'bicycle',
    'car',
    'motorcycle',
    'airplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic_light',
    'fire_hydrant',
    'stop_sign',
    'parking_meter',
    'bench',
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    'backpack',
    'umbrella',
    'handbag',
    'tie',
    'suitcase',
    'frisbee',
    'skis',
    'snowboard',
    'sports_ball',
    'kite',
    'baseball_bat',
    'baseball_glove',
    'skateboard',
    'surfboard',
    'tennis_racket',
    'bottle',
    'wine_glass',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'hot_dog',
    'pizza',
    'donut',
    'cake',
    'chair',
    'couch',
    'potted_plant',
    'bed',
    'dining_table',
    'toilet',
    'tv',
    'laptop',
    'mouse',
    'remote',
    'keyboard',
    'cell_phone',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    'book',
    'clock',
    'vase',
    'scissors',
    'teddy_bear',
    'hair_drier',
    'toothbrush'
)
_object_name_to_id = {k: v for v, k in enumerate(_object_id_to_name)}
HicoDet.objects = tuple(_object_id_to_name)

_predicate_id_to_name = [
    'adjust',
    'assemble',
    'block',
    'blow',
    'board',
    'break',
    'brush_with',
    'buy',
    'carry',
    'catch',
    'chase',
    'check',
    'clean',
    'control',
    'cook',
    'cut',
    'cut_with',
    'direct',
    'drag',
    'dribble',
    'drink_with',
    'drive',
    'dry',
    'eat',
    'eat_at',
    'exit',
    'feed',
    'fill',
    'flip',
    'flush',
    'fly',
    'greet',
    'grind',
    'groom',
    'herd',
    'hit',
    'hold',
    'hop_on',
    'hose',
    'hug',
    'hunt',
    'inspect',
    'install',
    'jump',
    'kick',
    'kiss',
    'lasso',
    'launch',
    'lick',
    'lie_on',
    'lift',
    'light',
    'load',
    'lose',
    'make',
    'milk',
    'move',
    'no_interaction',
    'open',
    'operate',
    'pack',
    'paint',
    'park',
    'pay',
    'peel',
    'pet',
    'pick',
    'pick_up',
    'point',
    'pour',
    'pull',
    'push',
    'race',
    'read',
    'release',
    'repair',
    'ride',
    'row',
    'run',
    'sail',
    'scratch',
    'serve',
    'set',
    'shear',
    'sign',
    'sip',
    'sit_at',
    'sit_on',
    'slide',
    'smell',
    'spin',
    'squeeze',
    'stab',
    'stand_on',
    'stand_under',
    'stick',
    'stir',
    'stop_at',
    'straddle',
    'swing',
    'tag',
    'talk_on',
    'teach',
    'text_on',
    'throw',
    'tie',
    'toast',
    'train',
    'turn',
    'type_on',
    'walk',
    'wash',
    'watch',
    'wave',
    'wear',
    'wield',
    'zip'
]
_predicate_name_to_id = {k: v for v, k in enumerate(_predicate_id_to_name)}
HicoDet.predicates = tuple(_predicate_id_to_name)
