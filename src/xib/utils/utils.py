from functools import wraps
from typing import Mapping, Sequence, Generator, Tuple, Any

import torch
import numpy as np


def noop(*_, **__):
    pass


def identity(*args):
    return args


def flatten_dict(input: Mapping, prefix: Sequence = ()) -> Generator[Tuple[Tuple, Any], None, None]:
    """Flatten a dictionary into a sequence of (tuple_key, value) tuples

    Example:
        >>> for k, v in flatten_dict({'a': 1, 'b': {'x': 2, 'y': 3}}):
        ...     print(k, v)
        ('a',) 1
        ('b', 'x') 2
        ('b', 'y') 3

    """
    for k, v in input.items():
        if isinstance(v, Mapping):
            yield from flatten_dict(v, prefix=(*prefix, k))
        else:
            yield (*prefix, k), v


def apples_to_apples(f):
    @wraps(f)
    def wrapper(cls, input):
        if isinstance(input, (int, str)):
            return f(cls, [input])[0]
        if isinstance(input, (torch.Tensor, np.ndarray)):
            if input.ndim == 0:
                return f(cls, [input.item()])[0]
            else:
                return np.array(f(cls, input))
        return f(cls, input)

    return wrapper
