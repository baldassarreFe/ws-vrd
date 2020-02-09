from typing import Tuple

import torch
from detectron2.structures import Boxes


def boxes_to_node_features(boxes: Boxes, image_size: Tuple[int, int]):
    """Compute node features from the bounding boxes of detected objects.

    - Elongation (height / width)
    - Elongation (width / height)
    - Area relative to the total image area

    Args:
        boxes:
        image_size: (height, width)

    Returns:

    """
    height, width = image_size

    # Boxes are represented as N x (x1, y1, x2, y2):
    x_min, y_min, x_max, y_max = boxes.tensor.unbind(dim=1)
    elongation = (y_max - y_min) / (x_max - x_min)

    relative_area = boxes.area() / (height * width)

    nodes = torch.stack([elongation, 1 / elongation, relative_area], dim=1)

    return nodes
