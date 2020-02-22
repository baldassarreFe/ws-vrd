from typing import Sequence, Union, Optional

import pandas as pd


class Vocabulary(object):
    def __init__(self, words: Sequence[str], name: Optional[str] = None):
        words = pd.Series(words)
        if not words.is_unique:
            raise ValueError("Vocabulary can not contain duplicate values")

        self._name = name if name is not None else "words"
        self._id_to_str = words.rename(self._name)
        self._str_to_id = pd.Series(
            self._id_to_str.index, index=self._id_to_str.values, name=self._name
        )

    @property
    def words(self):
        return self._id_to_str.tolist()

    @property
    def name(self):
        return self._name

    def __len__(self):
        return len(self._id_to_str)

    def get_str(self, id: Union[int, Sequence[int]]):
        return self._id_to_str.loc[id]

    def get_id(self, str: Union[str, Sequence[str]]):
        return self._str_to_id.loc[str]

    def __repr__(self):
        return f"{self.__class__.__name__}({self._name}={len(self)})"
