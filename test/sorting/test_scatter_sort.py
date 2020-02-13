import torch
import torch.testing

from xib.utils import scatter_sort


def test_scatter_sort(device):
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

    result_values, result_indexes = scatter_sort(src, index, descending=True)

    expected = torch.tensor([
        8.3, 2.3, 1.4,
        8.6, 7.6, 6.6, 6.4, 3.7, 1.7,
        9.1, 6.8,
        5.2, 2.0, 1.0, 0.4
    ], device=device)

    torch.testing.assert_allclose(result_values, expected)
    torch.testing.assert_allclose(src[result_indexes], expected)
