"""
Module for visual feature extraction: given an image we want to extract visual features of objects.

Visual features are extracted using RoI pooling on top of the feature pyramid produced by a detectron model.
Objects bounding boxes can come both from ground-truth annotations and from detectron's own detections.
The detectron model can be either a pretrained model from the model zoo, or a custom model.

Furthermore, two fully-connected graphs are created, resp. for ground-truth objects and detected objects. A set of
linear features is attached to nodes and the edges of this graph, to describe geometric properties of object boxes
and pairs of objects.
"""

from .wrapper import DetectronWrapper
from .node_features import boxes_to_node_features
from .edge_features import boxes_to_edge_features

__all__ = [
    'DetectronWrapper',
    'boxes_to_node_features',
    'boxes_to_edge_features',
]
