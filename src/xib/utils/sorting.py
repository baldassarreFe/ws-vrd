from typing import Tuple, Optional

import torch
from torch import Tensor, LongTensor


def scatter_sort(src: Tensor, index: LongTensor, descending=False, dim_size=None,
                 out: Optional[Tuple[Tensor, LongTensor]] = None) -> Tuple[Tensor, LongTensor]:
    if src.ndimension() > 1:
        raise ValueError('Only implemented for 1D tensors')

    if dim_size is None:
        dim_size = index.max() + 1

    if out is None:
        result_values = torch.empty_like(src)
        result_indexes = index.new_empty(src.shape)
    else:
        result_values, result_indexes = out

    sizes = index.new_zeros(dim_size).scatter_add_(dim=0, index=index, src=torch.ones_like(index)).tolist()

    start = 0
    for size in sizes:
        end = start + size
        values, indexes = torch.sort(src[start:end], dim=0, descending=descending)
        result_values[start:end] = values
        result_indexes[start:end] = indexes + start
        start = end

    return result_values, result_indexes


def scatter_topk(src: Tensor, index: LongTensor, k: int, num_chunks=None, fill_value=None) \
        -> Tuple[Tensor, LongTensor, LongTensor]:
    if src.ndimension() > 1:
        raise ValueError('Only implemented for 1D tensors')

    if num_chunks is None:
        num_chunks = index.max().item() + 1

    if fill_value is None:
        fill_value = float('NaN')

    result_values = src.new_full((num_chunks, k), fill_value=fill_value)
    result_indexes_whole = index.new_full((num_chunks, k), fill_value=-1)
    result_indexes_within_chunk = index.new_full((num_chunks, k), fill_value=-1)

    chunk_sizes = index.new_zeros(num_chunks).scatter_add_(dim=0, index=index, src=torch.ones_like(index)).tolist()

    start = 0
    for chunk_idx, chunk_size in enumerate(chunk_sizes):
        chunk = src[start:start + chunk_size]
        values, indexes = torch.topk(chunk, k=min(k, chunk_size), dim=0)

        result_values[chunk_idx, :len(values)] = values
        result_indexes_within_chunk[chunk_idx, :len(indexes)] = indexes
        result_indexes_whole[chunk_idx, :len(indexes)] = indexes + start

        start += chunk_size

    return result_values, result_indexes_whole, result_indexes_within_chunk


def scatter_topk_2d_flat(src: Tensor, index: LongTensor, k: int, dim_size=None, fill_value=None) \
        -> Tuple[Tensor, Tuple[LongTensor, LongTensor], Tuple[LongTensor, LongTensor]]:
    """Finds the top k values in a 2D array partitioned along the dimension 0.

    ::

        +-----------------------+
        |          X            |
        |  X                    |
        |              X        |
        |     X                 |
        +-----------------------+
        |                       |
        |                 Y     |
        |       Y               |              +-------+
        |                       |              |X X X X|
        |                       |    top 4     +-------+
        |                       |  -------->   |X X X X|
        |                       |              +-------+
        |             Y         |              |Z Z Z Z|
        |                       |              +-------+
        |   Y                   |
        |                       |
        +-----------------------+
        |                       |
        |     Z       Z         |
        |                       |
        |        Z        Z     |
        |                       |
        +-----------------------+


    Args:
        src:
        index:
        k:
        dim_size:
        fill_value:

    Returns:

    """
    if src.ndimension() != 2:
        raise ValueError('Only implemented for 2D tensors')

    if dim_size is None:
        dim_size = index.max().item() + 1

    if fill_value is None:
        fill_value = float('NaN')

    ncols = src.shape[1]

    result_values = src.new_full((dim_size, k), fill_value=fill_value)
    result_indexes_whole_0 = index.new_full((dim_size, k), fill_value=-1)
    result_indexes_whole_1 = index.new_full((dim_size, k), fill_value=-1)
    result_indexes_within_chunk_0 = index.new_full((dim_size, k), fill_value=-1)
    result_indexes_within_chunk_1 = index.new_full((dim_size, k), fill_value=-1)

    chunk_sizes = index.new_zeros(dim_size).scatter_add_(dim=0, index=index, src=torch.ones_like(index)).tolist()

    start_src = 0
    for chunk_idx, chunk_size in enumerate(chunk_sizes):
        flat_chunk = src[start_src:start_src + chunk_size, :].flatten()
        flat_values, flat_indexes = torch.topk(flat_chunk, k=min(k, chunk_size * ncols), dim=0)
        result_values[chunk_idx, :len(flat_values)] = flat_values

        indexes_0 = flat_indexes / ncols
        indexes_1 = flat_indexes % ncols
        result_indexes_within_chunk_0[chunk_idx, :len(flat_indexes)] = indexes_0
        result_indexes_within_chunk_1[chunk_idx, :len(flat_indexes)] = indexes_1

        result_indexes_whole_0[chunk_idx, :len(flat_indexes)] = indexes_0 + start_src
        result_indexes_whole_1[chunk_idx, :len(flat_indexes)] = indexes_1

        start_src += chunk_size

    return (
        result_values,
        (result_indexes_whole_0, result_indexes_whole_1),
        (result_indexes_within_chunk_0, result_indexes_within_chunk_1)
    )
