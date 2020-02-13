from typing import Optional, Mapping, Any

import torch
from detectron2.structures import Instances

from .vocabulary import Vocabulary


def clone_instances(instances: Instances):
    return Instances(
        instances.image_size,
        **{
            k: v.clone()
            for k, v in instances.get_fields().items()
        }
    )


def instance_str(instances: Instances, vocabulary: Optional[Vocabulary] = None):
    if instances.has('classes'):
        if vocabulary is not None:
            classes = vocabulary.get_str(instances.classes.detach().cpu()).tolist()
        else:
            classes = [f'{l:d}' for l in instances.classes]
    else:
        classes = ['?'] * len(instances)

    if instances.has('boxes'):
        boxes = ['█'] * len(instances)
    else:
        boxes = ['?'] * len(instances)

    if instances.has('scores'):
        scores = [f'{s:.1%}' for s in instances.scores]
    else:
        scores = ['?'] * len(instances)

    return [f'({n} {b} {s})' for n, b, s in zip(classes, boxes, scores)]


def to_data_dict(instances: Instances, prefix=''):
    """Prepare a dict to build a pytorch_geometric.data.Data object"""
    def key_mapper(key: str) -> str:
        return '_'.join((prefix, key))
    res = {
        key_mapper(k): v for k, v in instances.get_fields().items()
        if k in {'classes', 'scores', 'linear_features', 'conv_features'}
    }
    if instances.has('boxes'):
        res[key_mapper('boxes')] = instances.boxes.tensor  # Nx4
    res[key_mapper('image_size')] = torch.tensor([instances.image_size])  # 1x2
    return res


def from_data_dict(data_dict: Mapping[str, Any], prefix=None):
    strip = len(prefix) + 1 if prefix is not None else 0
    return Instances(**{k[strip:]: v for k, v in data_dict.items()})
