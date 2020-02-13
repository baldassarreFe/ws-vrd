from __future__ import annotations

import dataclasses
from typing import Optional, Union, Iterator, Tuple, Sequence, Mapping, Any

import torch
from detectron2.structures import Boxes, Instances

from .boxes import matched_boxlist_union
from .instances import instance_str
from .vocabulary import Vocabulary


@dataclasses.dataclass
class VisualRelations(object):
    # @formatter:off
    relation_indexes:     torch.LongTensor
    relation_scores:      Optional[torch.Tensor] = None

    predicate_classes:    Optional[torch.LongTensor] = None
    predicate_scores:     Optional[torch.Tensor] = None

    conv_features:        Optional[torch.Tensor] = None
    linear_features:      Optional[torch.Tensor] = None

    # Not strictly related to a visual relation
    instances:            Optional[Instances] = None
    object_vocabulary:    Optional[Vocabulary] = None
    predicate_vocabulary: Optional[Vocabulary] = None
    # @formatter:on

    def __post_init__(self):
        if len(list(self._iter_fields())) == 0:
            raise ValueError('No field is set on this instance')

        devices = [self.relation_indexes.device] + [
            t.device for t in [
                self.relation_scores,
                self.predicate_classes, self.predicate_scores,
                self.conv_features, self.linear_features
            ] if t is not None
        ]

        lengths = [self.relation_indexes.shape[1]] + [
            len(t) for t in [
                self.relation_scores,
                self.predicate_classes, self.predicate_scores,
                self.conv_features, self.linear_features
            ] if t is not None
        ]

        if any(devices[0] != d for d in devices):
            raise ValueError('All tensors must be on the same device')
        if any(lengths[0] != l for l in lengths):
            raise ValueError('All tensors must have the same length')

    def _iter_fields(self) -> Iterator[Tuple[str, Union[torch.Tensor, Boxes]]]:
        """Iterate over non-null fields."""
        for k, v in self.__dict__.items():
            if v is not None:
                yield k, v

    @property
    def device(self):
        for _, v in self._iter_fields():
            if hasattr(v, 'device'):
                return v.device

    def predicate_str(self) -> Sequence[str]:
        if self.predicate_classes is not None:
            if self.predicate_vocabulary is not None:
                predicates = self.predicate_vocabulary.get_str(self.predicate_classes.detach().cpu()).tolist()
            else:
                predicates = [f'{p:d}' for p in self.predicate_classes]
        else:
            predicates = ['?'] * len(self)

        if self.predicate_scores is not None:
            scores = [f'{s:.1%}' for s in self.predicate_scores]
        else:
            scores = ['?'] * len(self)

        return [f'({p} {s})' for p, s in zip(predicates, scores)]

    def subject_str(self) -> Sequence[str]:
        if self.instances is not None:
            return instance_str(self.instances[self.relation_indexes[0]], self.object_vocabulary)
        else:
            return [f'#{s:d}' for s in self.relation_indexes[0]]

    def object_str(self) -> Sequence[str]:
        if self.instances is not None:
            return instance_str(self.instances[self.relation_indexes[1]], self.object_vocabulary)
        else:
            return [f'#{o:d}' for o in self.relation_indexes[1]]

    def relation_str(self):
        if self.relation_scores is not None:
            scores = [f'{s:.1%}' for s in self.relation_scores]
        else:
            scores = ['?'] * len(self)

        return [f'{{{s} {p} {o} {score}}}' for s, p, o, score
                in zip(self.subject_str(), self.predicate_str(), self.object_str(), scores)]

    def cpu(self) -> VisualRelations:
        return self.to('cpu')

    def cuda(self, device=Union[str, torch.device], non_blocking=False) -> VisualRelations:
        return self.to(device, non_blocking)

    def to(self, device=Union[str, torch.device], non_blocking=False, copy=False) -> VisualRelations:
        return dataclasses.replace(self, **{
            k: v.to(device, non_blocking, copy) for k, v in self._iter_fields()
        })

    def detach(self) -> VisualRelations:
        return dataclasses.replace(self, **{
            k: v.detach() for k, v in self._iter_fields()
        })

    def detach_(self) -> VisualRelations:
        for k, v in self._iter_fields():
            v.detach_()
        return self

    def subject_boxes(self) -> Boxes:
        return self.instances.boxes[self.relation_indexes[0]]

    def object_boxes(self) -> Boxes:
        return self.instances.boxes[self.relation_indexes[1]]

    def phrase_boxes(self):
        return matched_boxlist_union(self.subject_boxes(), self.object_boxes())

    def area_sum(self):
        return self.subject_boxes().area() + self.object_boxes().area()

    def area_union(self):
        return matched_boxlist_union(self.subject_boxes(), self.object_boxes()).area()

    def to_data_dict(self, prefix=''):
        """Prepare a dict to build a pytorch_geometric.data.Data object"""
        def key_mapper(key: str) -> str:
            if key.endswith('features'):
                key = f'relation_{key}'
            return '_'.join((prefix, key))
        return {
            key_mapper(k): v for k, v in self.__dict__.items()
            if k in {'relation_scores', 'relation_indexes',
                     'predicate_scores', 'predicate_classes',
                     'conv_features', 'linear_features'}
        }

    @classmethod
    def from_data_dict(cls, data_dict: Mapping[str, Any], prefix=''):
        strip = len(prefix) + 1 if prefix is not None else 0

        def key_mapper(key: str) -> str:
            key_strip = strip
            if key.endswith('features'):
                key_strip = strip + len('relation_')
            return key[key_strip:]

        return VisualRelations(**{key_mapper(k): v for k, v in data_dict.items()})

    def __len__(self):
        return self.relation_indexes.shape[1]

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> VisualRelations:
        return dataclasses.replace(self, **{
            k: v[item]
            for k, v in self._iter_fields()
        })

    def __iter__(self) -> Iterator[VisualRelations]:
        for i in range(len(self)):
            yield self[i]

    def __str__(self):
        fields = ', '.join(k for k, v in self._iter_fields())
        return f'{self.__class__.__name__}({len(self)}, [{fields}])'

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items()
                if k not in {'object_vocabulary', 'predicate_vocabulary'}}
