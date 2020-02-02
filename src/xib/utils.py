import signal
import random
import inspect
import importlib

from functools import wraps
from typing import Callable, Mapping, Sequence, Generator, Tuple, Any

import torch
import numpy as np
import namesgenerator

from loguru import logger


def random_name():
    nice_name = namesgenerator.get_random_name()
    random_letters = ''.join(chr(random.randint(ord('A'), ord('Z'))) for _ in range(6))
    return nice_name + '_' + random_letters


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


def import_(fullname):
    package, name = fullname.rsplit('.', maxsplit=1)
    package = importlib.import_module(package)
    return getattr(package, name)


def check_extra_parameters(f, kwargs):
    signature = inspect.signature(f)
    if not set(kwargs.keys()).issubset(set(signature.parameters.keys())):
        logger.warning(f'Extra parameters found for {f}, '
                       f'expected {{{set(signature.parameters.keys())}}}, '
                       f'given {{{set(kwargs.keys())}}}')


def apples_to_apples(f):
    @wraps(f)
    def wrapper(cls, input):
        if isinstance(input, (int, str)):
            return f(cls, [input])[0]
        if isinstance(input, (torch.Tensor, np.ndarray)):
            return np.array(f(cls, input))
        return f(cls, input)

    return wrapper


class SigIntHandler(object):
    def __init__(self, handler: Callable):
        self.handler = handler
        self.old_handler = signal.signal(signal.SIGINT, self._internal_handler)

    def _internal_handler(self, sig, frame):
        signal.signal(signal.SIGINT, self.old_handler)
        logger.warning('Received SIGINT.')
        self.handler()
        self.old_handler = signal.signal(signal.SIGINT, self._internal_handler)


class SigIntCatcher(SigIntHandler):
    def __init__(self):
        self._caught = False
        super().__init__(self._handler)

    def _handler(self):
        self._caught = True

    def __bool__(self):
        return self._caught
