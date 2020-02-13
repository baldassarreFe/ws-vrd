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


def scatter_topk(src: Tensor, index: LongTensor, k: int, dim_size=None, fill_value=None) -> Tuple[Tensor, LongTensor]:
    if src.ndimension() > 1:
        raise ValueError('Only implemented for 1D tensors')

    if dim_size is None:
        dim_size = index.max().item() + 1

    if fill_value is None:
        fill_value = float('NaN')

    result_values = src.new_full((dim_size * k,), fill_value=fill_value)
    result_indexes = index.new_full((dim_size * k,), fill_value=-1)

    sizes = index.new_zeros(dim_size).scatter_add_(dim=0, index=index, src=torch.ones_like(index)).tolist()

    start_src = 0
    start_res = 0
    for size in sizes:
        values, indexes = torch.topk(src[start_src:start_src + size], k=min(k, size), dim=0)
        result_values[start_res:start_res + len(values)] = values
        result_indexes[start_res:start_res + len(indexes)] = indexes + start_src
        start_src += size
        start_res += k

    return result_values, result_indexes
