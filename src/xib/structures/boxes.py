from typing import overload

import torch
from detectron2.structures import Boxes


def area_intersection(boxes_a: Boxes, boxes_b: Boxes):
    if len(boxes_a) != len(boxes_b):
        raise ValueError(f"The two boxes must have the same length")

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


def matched_boxlist_intersection(boxes_a: Boxes, boxes_b: Boxes) -> Boxes:
    """ Compute the intersection box between corresponding boxes:

    If the intersection is empty, a (0, 0, 0, 0) box is returned.

    ::

      0------------------------------------------->
      |                                           x
      |   x1, y1 -------------------+
      |     |                       |
      |     |    a                  |
      |     |                       |
      |     |                       |
      |     |        x1, y1 --------|--------+
      |     |          |            |        |
      |     |          |   inters   |        |
      |     |          |            |        |
      |     +-------------------- x2, y2     |
      |                |                     |
      |                |                 b   |
      |                +------------------ x2, y2
      V y

    Args:
        boxes_a:
        boxes_b:

    Returns:

    """
    if len(boxes_a) != len(boxes_b):
        raise ValueError(f"The two boxes must have the same length")

    # Boxes are represented as N x (x1, y1, x2, y2):
    a = boxes_a.tensor  # N x 4
    b = boxes_b.tensor  # N x 4

    inters_top_left = torch.max(a[:, :2], b[:, :2])  # N x (x1, y1)
    inters_bottom_right = torch.min(a[:, 2:], b[:, 2:])  # N x (x2, y2)
    boxes_inters = torch.cat((inters_top_left, inters_bottom_right), dim=1)

    # If height or width are negative, the box is empty
    empty = torch.any(inters_top_left >= inters_bottom_right, dim=1)
    boxes_inters[empty] = 0

    return Boxes(boxes_inters)


@overload
def matched_boxlist_union(a: Boxes, b: Boxes) -> Boxes:
    ...


@overload
def matched_boxlist_union(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ...


def matched_boxlist_union(a, b):
    """ Compute the union box between corresponding boxes:

    ::

      0------------------------------------>     0------------------------------------>
      |                                    x     |                                    x
      |   x1, y1 ------------+                   |   u1, u1 ---------------------+
      |     |                |                   |     |                         |
      |     |     x1, y1 ----|--------+      =>  |     |                         |
      |     | a     |        |        |          |     |                         |
      |     +------------- x2, y2     |          |     |                         |
      |             |             b   |          |     |                         |
      |             +-------------- x2, y2       |     +---------------------- u2, u2
      V y                                        V y

    Args:
        a: [N, 4] tensor or Boxes instance
        b: [N, 4] tensor or Boxes instance

    Returns:

    """
    if isinstance(a, Boxes) and isinstance(b, Boxes):
        return Boxes(matched_boxlist_union(a.tensor, b.tensor))

    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        raise ValueError(f"Unkown data type {type(a)}, {type(b)}")

    if len(a) != len(b):
        raise ValueError(f"The two boxes must have the same length")
    if not a.shape[1] == b.shape[1] == 4:
        raise ValueError(f"Wrong box shape {a.shape}, {b.shape}")

    # Boxes are represented as N x (x1, y1, x2, y2):
    union_top_left = torch.min(a[:, :2], b[:, :2])  # N x (x1, y1)
    union_bottom_right = torch.max(a[:, 2:], b[:, 2:])  # N x (x2, y2)
    boxes_union = torch.cat((union_top_left, union_bottom_right), dim=1)

    return boxes_union
