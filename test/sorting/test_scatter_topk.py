import torch
import torch.testing

from xib.utils import scatter_topk


def test_scatter_topk(device):
    src = torch.tensor([
        2.3, 8.3, 1.4,
        3.7, 6.6, 1.7, 8.6, 6.4, 7.6,
        6.8, 9.1,
        5.2, 2.0, 0.4, 1.0
    ], device=device)
    index = torch.tensor([
        0, 0, 0,
        1, 1, 1, 1, 1, 1,
        2, 2,
        3, 3, 3, 3
    ], device=device)

    result_values, result_indexes = scatter_topk(src, index, k=4)

    expected_values = torch.tensor([
        8.3, 2.3, 1.4, float('NaN'),
        8.6, 7.6, 6.6, 6.4,
        9.1, 6.8, float('NaN'), float('NaN'),
        5.2, 2.0, 1.0, 0.4
    ], device=device)
    expected_indexes = torch.tensor([
        1, 0, 2, -1,
        6, 8, 4, 7,
        10, 9, -1, -1,
        11, 12, 14, 13
    ], device=device)

    torch.testing.assert_allclose(result_values, expected_values, equal_nan=True)
    torch.testing.assert_allclose(result_indexes, expected_indexes)
