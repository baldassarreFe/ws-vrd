from __future__ import annotations

import dataclasses
from typing import Optional, Union, Iterator, Tuple

import torch
from .boxes import Boxes, area_intersection


@dataclasses.dataclass(frozen=True)
class VisualRelations(object):
    # @formatter:off
    subject_classes:   Optional[torch.LongTensor] = None
    predicate_classes: Optional[torch.LongTensor] = None
    object_classes:    Optional[torch.LongTensor] = None

    subject_scores:    Optional[torch.FloatTensor] = None
    predicate_scores:  Optional[torch.FloatTensor] = None
    object_scores:     Optional[torch.FloatTensor] = None

    subject_logits:    Optional[torch.FloatTensor] = None
    predicate_logits:  Optional[torch.FloatTensor] = None
    object_logits:     Optional[torch.FloatTensor] = None

    subject_boxes:    Optional[Boxes] = None
    object_boxes:     Optional[Boxes] = None
    # @formatter:on

    def __post_init__(self):
        devices = []
        lengths = []
        for _, v in self._iter_fields():
            devices.append(v.device)
            lengths.append(len(v))

        if len(lengths) == 0:
            raise ValueError('No field is set on this instance')
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
            return v.device

    def to(self, device=Union[str, torch.device]) -> VisualRelations:
        return dataclasses.replace(self, **{
            k: v.to(device) for k, v in self._iter_fields()
        })

    def cat(self, other: VisualRelations) -> VisualRelations:
        def _cat(a, b):
            if a is None and b is None:
                return None
            if isinstance(a, torch.Tensor):
                return torch.cat((a, b), dim=0)
            if isinstance(a, Boxes):
                return Boxes.cat([a, b])
            raise TypeError(f'Can not concatenate {type(a)} and {type(b)}')

        return dataclasses.replace(self, **{
            k: _cat(self.__dict__[k], other.__dict__[k])
            for k in set.union(set(self.__dict__.keys()), set(other.__dict__.keys()))
        })
    
    def area_sum(self):
        return self.subject_boxes.area() + self.object_boxes.area()

    def area_union(self):
        return self.areas_sum() - area_intersection(self.subject_boxes, self.object_boxes)

    def __len__(self):
        for _, v in self._iter_fields():
            return len(v)

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> VisualRelations:
        return dataclasses.replace(self, **{
            k: v[item]
            for k, v in self._iter_fields()
        })

    def __iter__(self) -> Iterator[VisualRelations]:
        for i in range(len(self)):
            yield self[i]
