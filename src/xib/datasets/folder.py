from pathlib import Path
from typing import Sequence, Union, Optional

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm


class DatasetFolder(object):
    def __init__(self, paths: Sequence[Path]):
        self.paths = paths
        self.samples = [None] * len(paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        sample = self.samples[item]
        if sample is None:
            sample = torch.load(self.paths[item])
        return sample

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def split(self, percent: float, *, random_state: Optional[int] = None):
        rg = np.random.default_rng(random_state)
        indexes = rg.permutation(len(self))
        split_1 = indexes[:int(percent * len(indexes))]
        split_2 = indexes[int(percent * len(indexes)):]
        return DatasetFolderSubset(self, split_1), DatasetFolderSubset(self, split_2)

    def load_eager(self, *, subset: Optional[Sequence[int]] = None, quiet=False):
        if subset is None:
            subset = range(len(self))
        with tqdm(subset, desc='Loading', unit='s', disable=quiet) as bar:
            for i in bar:
                if self.samples[i] is None:
                    self.samples[i] = torch.load(self.paths[i])
        logger.info(f'Loaded {len(subset)} samples in {bar.last_print_t - bar.start_t:.1f}s')

    @classmethod
    def from_folder(cls, folder: Union[str, Path], *, suffix: str):
        folder = Path(folder).expanduser().resolve()
        paths = sorted(p for p in folder.iterdir() if p.name.endswith(suffix))
        if len(paths) == 0:
            logger.warning(f'Empty folder: no {suffix} file found in {folder}')
        return cls(paths)


class DatasetFolderSubset(object):
    def __init__(self, df: DatasetFolder, indexes: Sequence[int]):
        self.df = df
        self.indexes = indexes

    def __getitem__(self, item):
        return self.df[self.indexes[item]]

    def __len__(self):
        return len(self.indexes)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def load_eager(self, quiet=False):
        self.df.load_eager(subset=self.indexes, quiet=quiet)
