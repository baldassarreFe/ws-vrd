from .boxes import Boxes, area_intersection
from .relations import VisualRelations

from detectron2.structures import Instances

__all__ = [
    'Boxes',
    'area_intersection',
    'VisualRelations',
    'Instances',
]
