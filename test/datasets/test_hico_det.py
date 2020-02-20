import torch
import numpy as np

from xib.datasets import VrDataset


def test_mappings():
    for ids in [5, [1, 2, 3], np.array([4, 5, 6]), torch.tensor([7, 8, 9])]:
        names = VrDataset.object_id_to_name(ids)
        ids_back = VrDataset.object_name_to_id(names)
        np.testing.assert_array_equal(ids, ids_back)

    for names in ['boat', ['car', 'person', 'airplane'], np.array(['car', 'car', 'boat'])]:
        ids = VrDataset.object_name_to_id(names)
        names_back = VrDataset.object_id_to_name(ids)
        np.testing.assert_array_equal(names, names_back)

    for ids in [5, [1, 2, 3], np.array([4, 5, 6]), torch.tensor([7, 8, 9])]:
        names = VrDataset.predicate_id_to_name(ids)
        ids_back = VrDataset.predicate_name_to_id(names)
        np.testing.assert_array_equal(ids, ids_back)

    for names in ['eat', ['sip', 'zip', 'tie'], np.array(['tie', 'tie', 'eat'])]:
        ids = VrDataset.predicate_name_to_id(names)
        names_back = VrDataset.predicate_id_to_name(ids)
        np.testing.assert_array_equal(names, names_back)
