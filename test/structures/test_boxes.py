import torch
import torch.testing
from detectron2.structures import Boxes

from xib.structures import area_intersection


def test_area_intersection(device):
    boxes_a = Boxes(torch.tensor(
        [(5, 5, 10, 10)] * 14,
        device=device, dtype=torch.float)
    )

    boxes_b = Boxes(torch.tensor([
        # Outside
        (1, 1, 3, 3),
        (11, 11, 13, 13),
        (1, 11, 3, 13),
        (11, 1, 13, 3),

        # Fully inside
        (6, 7, 9, 9),
        (5, 5, 10, 10),

        # Partial overlap by one corner
        (1, 1, 6, 7),
        (9, 9, 13, 13),
        (1, 7, 7, 13),
        (8, 1, 13, 6),

        # Partial overlap by 2 corners
        (1, 1, 13, 7),
        (1, 7, 13, 13),
        (1, 1, 8, 13),
        (8, 1, 13, 13),
    ], device=device, dtype=torch.float))

    expected_areas = torch.tensor([
        0,
        0,
        0,
        0,
        6,
        25,
        2,
        1,
        6,
        2,
        10,
        15,
        15,
        10,
    ], device=device, dtype=torch.float)

    areas = area_intersection(boxes_a, boxes_b)
    torch.testing.assert_allclose(areas, expected_areas)
