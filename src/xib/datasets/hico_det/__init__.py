from .sample import HicoDetSample
from .dataset import HicoDet
from .metadata import OBJECTS, PREDICATES
from .catalog import register_hico

from ..catalog import DatasetCatalog

DatasetCatalog.register('hico_det', {
    'class': HicoDet,
    'metadata': {
        'objects': OBJECTS,
        'predicates': PREDICATES
    },
})

__all__ = [
    'HicoDetSample',
    'HicoDet',
    'OBJECTS',
    'PREDICATES',
]
