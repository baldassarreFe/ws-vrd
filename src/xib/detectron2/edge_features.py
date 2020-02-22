from typing import Tuple

import torch
import numpy as np
from detectron2.structures import Boxes, pairwise_iou


def boxes_to_edge_features(boxes: Boxes, image_size: Tuple[int, int]):
    """Compute pairwise edge features from the bounding boxes of detected objects.

    - Euclidean distance between box centers relative to sqrt(area) of the image
    - Sin and cos of the delta between box centers
    - Intersection over union
    - Relative area of the first box w.r.t. the second box

    Args:
        boxes:
        image_size: (height, width)

    Returns:

    """
    height, width = image_size

    # Boxes are represented as N x (x1, y1, x2, y2):
    N = len(boxes)
    indices = torch.from_numpy(np.indices((N, N)).reshape(2, -1))  # 2 x (N*N)

    centers = boxes.get_centers()  # N x 2
    areas = boxes.area()  # N

    # delta[i, j] = centers[j] - centers[i]
    delta = centers[None, :, :] - centers[:, None, :]  # N x N x 2
    delta = delta.view(N * N, 2)  # N*N x 2
    relative_dist = delta.norm(dim=1) / np.sqrt(height * width)  # N*N
    angles = torch.atan2(delta[:, 1], delta[:, 0])  # N*N
    sin = torch.sin(angles)  # N*N
    cos = torch.cos(angles)  # N*N

    def quantize_angles(angles):
        """Quantize angles in the ranges 45-135, 135-225, 225-315, 315-45"""
        angles = angles - np.pi / 4
        top_half = torch.sin(angles) >= 0
        right_half = torch.sin(angles) >= 0

        result = torch.empty(len(angles), 4, dtype=torch.bool)
        result[:, 0] = top_half & right_half
        result[:, 1] = top_half & ~right_half
        result[:, 2] = ~top_half & ~right_half
        result[:, 3] = ~top_half & right_half

        return result.float()

    quadrants = quantize_angles(angles)

    iou = pairwise_iou(boxes, boxes)  # N x N
    iou = iou.view(N * N)  # N*N

    # relative_area[i, j] = area[i] / area[j]
    relative_area = areas[:, None] / areas[None, :]  # N x N
    relative_area = relative_area.view(N * N)  # N*N

    features = torch.stack(
        [
            relative_dist,
            sin,
            cos,
            *quadrants.unbind(dim=1),
            iou,
            relative_area,
            1 / relative_area,
        ],
        dim=1,
    )  # N x num_feats

    # Remove elements on the diagonal (i.e. self-relationships)
    mask = indices[0] != indices[1]
    features = features[mask]  # (N*N - N) x 4
    indices = indices[:, mask]  # 2 x (N*N -1)

    return features, indices
