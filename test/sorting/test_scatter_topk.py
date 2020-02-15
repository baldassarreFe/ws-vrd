import torch
import torch.testing

from xib.utils import scatter_topk_1d, scatter_topk_2d_flat, scatter_topk


def test_scatter_topk_1d(device):
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

    result_values, result_indexes_whole, result_indexes_within = scatter_topk_1d(src, index, k=4)

    expected_values = torch.tensor([
        [8.3, 2.3, 1.4, float('NaN')],
        [8.6, 7.6, 6.6, 6.4],
        [9.1, 6.8, float('NaN'), float('NaN')],
        [5.2, 2.0, 1.0, 0.4],
    ], device=device)
    expected_indexes_whole = torch.tensor([
        [ 1,  0,  2, -1],
        [ 6,  8,  4,  7],
        [10,  9, -1, -1],
        [11, 12, 14, 13],
    ], device=device)
    expected_indexes_within = torch.tensor([
        [1,  0,  2, -1],
        [3,  5,  1,  4],
        [1,  0, -1, -1],
        [0,  1,  3,  2],
    ], device=device)

    torch.testing.assert_allclose(result_values, expected_values, equal_nan=True)
    torch.testing.assert_allclose(result_indexes_whole, expected_indexes_whole)
    torch.testing.assert_allclose(result_indexes_within, expected_indexes_within)


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

    result_values, result_indexes_whole, result_indexes_within = scatter_topk(src, index, k=4)

    expected_values = torch.tensor([
        8.3, 2.3, 1.4, float('NaN'),
        8.6, 7.6, 6.6, 6.4,
        9.1, 6.8, float('NaN'), float('NaN'),
        5.2, 2.0, 1.0, 0.4,
    ], device=device)
    expected_indexes_whole = torch.tensor([
         1,  0,  2, -1,
         6,  8,  4,  7,
        10,  9, -1, -1,
        11, 12, 14, 13,
    ], device=device)
    expected_indexes_within = torch.tensor([
        1,  0,  2, -1,
        3,  5,  1,  4,
        1,  0, -1, -1,
        0,  1,  3,  2,
    ], device=device)

    torch.testing.assert_allclose(result_values, expected_values, equal_nan=True)
    torch.testing.assert_allclose(result_indexes_whole, expected_indexes_whole)
    torch.testing.assert_allclose(result_indexes_within, expected_indexes_within)


def test_scatter_topk_2d(device):
    src = torch.tensor([
        [4.0, 1.7],
        [6.4, 6.1],
        [5.6, 7.8],

        [7.8, 5.5],
        [0.1, 3.6],
        [7.4, 9.7],
        [0.9, 4.3],
        [4.5, 7.6],

        [3.1, 2.3],
        [1.0, 4.9]
    ], device=device)
    index = torch.tensor([
        0, 0, 0,
        1, 1, 1, 1, 1,
        2, 2,
    ], device=device)

    result_values, result_indexes_whole, result_indexes_within = scatter_topk_2d_flat(src, index, k=5)

    expected_values = torch.tensor([
        [7.8, 6.4, 6.1, 5.6, 4.0],
        [9.7, 7.8, 7.6, 7.4, 5.5],
        [4.9, 3.1, 2.3, 1.0, float('NaN')]
    ], device=device)
    expected_indexes_whole_0 = torch.tensor([
        [2, 1, 1, 2, 0],
        [5, 3, 7, 5, 3],
        [9, 8, 8, 9, -1]
    ], device=device)
    expected_indexes_whole_1 = torch.tensor([
        [1, 0, 1, 0, 0],
        [1, 0, 1, 0, 1],
        [1, 0, 1, 0, -1]
    ], device=device)

    expected_indexes_within_0 = torch.tensor([
        [2, 1, 1, 2, 0],
        [2, 0, 4, 2, 0],
        [1, 0, 0, 1, -1]
    ], device=device)
    expected_indexes_within_1 = torch.tensor([
        [1, 0, 1, 0, 0],
        [1, 0, 1, 0, 1],
        [1, 0, 1, 0, -1]
    ], device=device)

    torch.testing.assert_allclose(result_values, expected_values, equal_nan=True)
    torch.testing.assert_allclose(result_indexes_whole[0], expected_indexes_whole_0)
    torch.testing.assert_allclose(result_indexes_whole[1], expected_indexes_whole_1)
    torch.testing.assert_allclose(result_indexes_within[0], expected_indexes_within_0)
    torch.testing.assert_allclose(result_indexes_within[1], expected_indexes_within_1)
