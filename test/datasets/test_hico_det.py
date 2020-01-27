import torch
import numpy as np

from xib.datasets import HicoDet


def test_mappings():
    for ids in [5, [1, 2, 3], np.array([4, 5, 6]), torch.tensor([7, 8, 9])]:
        names = HicoDet.object_id_to_name(ids)
        ids_back = HicoDet.object_name_to_id(names)
        np.testing.assert_array_equal(ids, ids_back)

    for names in ['boat', ['car', 'person', 'airplane'], np.array(['car', 'car', 'boat'])]:
        ids = HicoDet.object_name_to_id(names)
        names_back = HicoDet.object_id_to_name(ids)
        np.testing.assert_array_equal(names, names_back)

    for ids in [5, [1, 2, 3], np.array([4, 5, 6]), torch.tensor([7, 8, 9])]:
        names = HicoDet.predicate_id_to_name(ids)
        ids_back = HicoDet.predicate_name_to_id(names)
        np.testing.assert_array_equal(ids, ids_back)

    for names in ['eat', ['sip', 'zip', 'tie'], np.array(['tie', 'tie', 'eat'])]:
        ids = HicoDet.predicate_name_to_id(names)
        names_back = HicoDet.predicate_id_to_name(ids)
        np.testing.assert_array_equal(names, names_back)
