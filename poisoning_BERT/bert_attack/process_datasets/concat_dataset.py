"""
concatenate multiple datasets
"""


import bisect

import numpy as np
from typing import List

from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate


class ConcatDataset(Dataset):
    """
    datasets: list of pytorch datasets
    sample_ratio: list of int, determine sample ratio
    """
    @staticmethod
    def cumsum(datasets: List[Dataset], sample_ratios: List[int]) -> List[int]:
        """返回多个datasets在concat后,每个dataset的end_idx"""
        r, s = [], 0
        for dataset, ratio in zip(datasets, sample_ratios):
            curr_len = int(ratio * len(dataset))
            r.append(curr_len + s)
            s += curr_len
        return r

    def __init__(self, datasets, sample_ratios=1):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, "datasets should not be an empty iterable"
        self.datasets = list(datasets)
        if isinstance(sample_ratios, int):
            sample_ratios = [sample_ratios] * len(self.datasets)
        self.sample_ratios = sample_ratios
        self.cumulative_sizes = self.cumsum(self.datasets, sample_ratios)
        self.real_sizes = [len(d) for d in self.datasets]

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx, sample_idx = self._get_dataset_and_sample_index(idx)
        return self.datasets[dataset_idx][sample_idx]

    def _get_dataset_and_sample_index(self, idx: int):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        sample_idx = sample_idx % self.real_sizes[dataset_idx]
        return dataset_idx, sample_idx

    def collater(self, samples):
        # For now only supports datasets with same underlying collater implementations
        if hasattr(self.datasets[0], 'collater'):
            return self.datasets[0].collater(samples)
        else:
            return default_collate(samples)

    def size(self, idx: int):
        """
        Return an example's size as a float or tuple.
        """
        dataset_idx, sample_idx = self._get_dataset_and_sample_index(idx)
        return self.datasets[dataset_idx].size(sample_idx)

    def num_tokens(self, index: int):
        return np.max(self.size(index))

    def attr(self, attr: str, index: int):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, index)
        return getattr(self.datasets[dataset_idx], attr, None)

    @property
    def sizes(self):
        _dataset_sizes = []
        for ds, sr in zip(self.datasets, self.sample_ratios):
            if isinstance(ds.sizes, np.ndarray):
                _dataset_sizes.append(np.tile(ds.sizes, sr))
            else:
                # Only support underlying dataset with single size array.
                assert isinstance(ds.sizes, list)
                _dataset_sizes.append(np.tile(ds.sizes[0], sr))
        return np.concatenate(_dataset_sizes)

    def __getattr__(self, item):
        """other dataset func"""
        return getattr(self.datasets[0], item)
