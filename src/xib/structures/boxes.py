from typing import Tuple

import torch
import numpy as np
from detectron2.structures import Boxes, pairwise_iou


def area_intersection(boxes_a: Boxes, boxes_b: Boxes):
    if len(boxes_a) != len(boxes_b):
        raise ValueError(f'The two boxes must have the same length')

    # Boxes are represented as N x (x1, y1, x2, y2):
    a = boxes_a.tensor  # N x 4
    b = boxes_b.tensor  # N x 4

    # Compute the coordinates of the intersection box:
    #
    #  x1, y1 -------------------+
    #    |                       |
    #    |    a                  |
    #    |                       |
    #    |                       |
    #    |        x1, y1 --------|--------+
    #    |          |            |        |
    #    |          |   inters   |        |
    #    |          |            |        |
    #    +-------------------- x2, y2     |
    #               |                     |
    #               |                 b   |
    #               +------------------ x2, y2
    #
    inters_top_left = torch.max(a[:, :2], b[:, :2])  # N x (x1, y1)
    inters_bottom_right = torch.min(a[:, 2:], b[:, 2:])  # N x (x2, y2)

    # If height or width are negative, the box is empty
    width_height = (inters_bottom_right - inters_top_left).clamp(min=0)
    area = width_height[:, 0] * width_height[:, 1]

    return area


def boxes_to_node_features(boxes: Boxes, size=Tuple[int, int]):
    """Compute node features from the bounding boxes of detected objects.

    - Elongation (height / width)
    - Area relative to the total image area

    Args:
        boxes:
        size: (height, width)

    Returns:

    """
    height, width = size

    # boxes is a N x 4 tensor
    x_min, y_min, x_max, y_max = boxes.tensor.unbind(dim=1)
    elongation = (y_max - y_min) / (x_max - x_min)

    area = boxes.area() / (height * width)

    nodes = torch.stack([elongation, area], dim=1)

    return nodes


def boxes_to_edge_features(boxes: Boxes, size=Tuple[int, int]):
    """Compute pairwise edge features from the bounding boxes of detected objects.

    - Euclidean distance between box centers relative to sqrt(area) of the image
    - Sin and cos of the delta between box centers
    - Intersection over union
    - Relative area of the first box w.r.t. the second box

    Args:
        boxes:
        size: (height, width)

    Returns:

    """
    height, width = size

    # boxes is a N x 4 tensor
    N = len(boxes)
    indices = torch.from_numpy(np.indices((N, N)).reshape(2, -1))  # 2 x (N*N)

    centers = boxes.get_centers()  # N x 2
    areas = boxes.area()  # N

    # delta[i, j] = centers[j] - centers[i]
    delta = centers[None, :, :] - centers[:, None, :]  # N x N x 2
    delta = delta.view(N * N, 2)  # N*N x 2
    distances = delta.norm(dim=1) / np.sqrt(height * width)  # N*N
    # TODO quantize angle in 45-135, 135-225, 225-315, 315-45
    angles = torch.atan2(delta[:, 1], delta[:, 0])  # N*N
    sin = torch.sin(angles)  # N*N
    cos = torch.cos(angles)  # N*N

    iou = pairwise_iou(boxes, boxes)  # N x N
    iou = iou.view(N * N)  # N*N

    # relative_area[i, j] = area[i] / area[j]
    relative_area = areas[:, None] / areas[None, :]  # N x N
    relative_area = relative_area.view(N * N)  # N*N

    edges = torch.stack([distances, sin, cos, iou, relative_area], dim=1)  # N x num_feats

    # Remove elements on the diagonal (i.e. self-relationships)
    mask = indices[0] != indices[1]
    edges = edges[mask]  # (N*N - N) x 4
    indices = indices[:, mask]  # 2 x (N*N -1)

    return edges, indices
