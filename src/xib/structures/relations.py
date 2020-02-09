from __future__ import annotations

import dataclasses
from typing import Optional, Union, Iterator, Tuple

import torch
from detectron2.structures import Boxes, Instances

from .boxes import matched_boxlist_union


@dataclasses.dataclass
class VisualRelations(object):
    # @formatter:off
    relation_indexes:  torch.LongTensor
    predicate_classes: Optional[torch.LongTensor] = None
    predicate_scores:  Optional[torch.Tensor] = None

    conv_features:    Optional[torch.Tensor] = None
    linear_features:  Optional[torch.Tensor] = None

    instances:         Optional[Instances] = None
    # @formatter:on

    def __post_init__(self):
        if len(list(self._iter_fields())) == 0:
            raise ValueError('No field is set on this instance')

        devices = []
        lengths = []
        for name, v in self._iter_fields():
            if name == 'instances':
                devices.append(v.boxes.device)
            elif name == 'relation_indexes':
                devices.append(v.device)
                lengths.append(v.shape[1])
            else:
                devices.append(v.device)
                lengths.append(len(v))

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

    def cat(self, other: VisualRelations) -> VisualRelations:
        def _cat(a, b):
            if a is None and b is None:
                return None
            if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                return torch.cat((a, b), dim=0)
            if isinstance(a, Instances) and isinstance(b, Instances):
                return Instances.cat((a, b))
            raise TypeError(f'Can not concatenate {type(a)} and {type(b)}')

        return dataclasses.replace(self, **{
            k: _cat(self.__dict__[k], other.__dict__[k])
            for k in set.union(set(self.__dict__.keys()), set(other.__dict__.keys()))
        })

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
