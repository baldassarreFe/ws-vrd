import time
import dataclasses
from enum import Enum
from pathlib import Path
from typing import Union, Iterable, Tuple, Dict, Optional, Callable

import torch
import torch.utils.data
from torch_geometric.data import Data

from loguru import logger
from detectron2.structures import Instances
from tqdm import tqdm

from ..utils import apples_to_apples
from ..structures import VisualRelations


@dataclasses.dataclass
class HicoDetSample(object):
    # Metadata
    filename: Union[Path, str]
    img_size: Tuple[int, int, int]

    # Ground-truth detections and visual relations
    gt_instances: Optional[Instances] = None
    gt_visual_relations: Optional[VisualRelations] = None
    gt_visual_relations_full: Optional[VisualRelations] = None

    # Input features: image features, detection boxes, box features
    d2_feature_pyramid: Optional[Dict[str, torch.Tensor]] = None
    d2_instances: Optional[Instances] = None
    d2_visual_relations_full: Optional[VisualRelations] = None


class HicoDet(torch.utils.data.Dataset):
    objects: Tuple[str]
    predicates: Tuple[str]

    class Mode(Enum):
        GT = 0
        D2 = 1

    def __init__(self, folder: Union[str, Path], mode: Mode, transforms: Optional[Callable] = None):
        self.mode = mode
        self.graphs = None
        self.transforms = transforms

        folder = Path(folder).expanduser().resolve()
        self.paths = sorted(p for p in folder.iterdir() if p.suffix == '.pth')
        if len(self.paths) == 0:
            logger.warning(f'Empty dataloader: no .pth file found in {folder}')

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
        if self.mode == HicoDet.Mode.GT:
            instances = sample.gt_instances
            visual_relations = sample.gt_visual_relations_full
        elif self.mode == HicoDet.Mode.D2:
            instances = sample.d2_instances
            visual_relations = sample.d2_visual_relations_full
        else:
            raise ValueError(f'Unknown mode: {self.mode}')

        edge_indexes = torch.stack((visual_relations.subject_indexes, visual_relations.object_indexes), dim=0)
        edge_linear_features = visual_relations.linear_features

        # target = binary encoding of the predicates
        target = torch.zeros(len(HicoDet.predicates), dtype=torch.float)
        target[visual_relations.predicate_classes] = 1.
        target.unsqueeze_(dim=1)

        graph = Data(
            num_nodes=len(instances),
            node_conv_features=instances.conv_features,
            node_linear_features=instances.linear_features,
            edge_index=edge_indexes,
            edge_attr=edge_linear_features,
            target=target,
        )

        return graph

    def load_eager(self):
        logger.debug(f'Starting to load {len(self.paths)} .pth files')
        vr_count = 0
        self.graphs = []
        start_time = time.perf_counter()
        for p in tqdm(self.paths, desc='Loading', unit='g'):
            g = self.make_graph(torch.load(p))
            vr_count += g.target.shape[1]
            self.graphs.append(g)
        logger.info(f'Loaded {vr_count:,} visual relations from '
                    f'{len(self.paths):,} .pth files in {time.perf_counter() - start_time:.1f}s')

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


# These are taken to be exactly the same and in the same order as COCO
# detectron2.data.catalog.MetadataCatalog.get('coco_2017_train').thing_classes
# except that there is a '_' instead of a ' ', to respect the annotation in the matlab file
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

# These are the predicates as they appear in the matlab file.
# They are sorted alphabetically since the order doesn't matter.
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
