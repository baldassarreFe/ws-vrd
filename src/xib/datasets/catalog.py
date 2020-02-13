from typing import Mapping, Any


class DatasetCatalog(object):
    _catalog = {}

    def __init__(self):
        raise ValueError('Do no create instances of this class')

    @staticmethod
    def register(name: str, dataset_dict: Mapping[str, Any]):
        DatasetCatalog._catalog[name] = dataset_dict

    @staticmethod
    def get(name) -> Mapping[str, Any]:
        return DatasetCatalog._catalog[name]