from pathlib import Path
from typing import Union

from .dataset import VrDataset
from .folder import DatasetFolder
from .hico_det import register_hico
from .vrd import register_vrd


def register_datasets(data_root: Union[str, Path]):
    register_hico(data_root)
    register_vrd(data_root)


__all__ = [
    'VrDataset',
    'DatasetFolder',
    'register_datasets',
]
