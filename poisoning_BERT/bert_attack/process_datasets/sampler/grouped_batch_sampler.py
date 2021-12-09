"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: multitas_sampler
@time: 2020/7/8 19:51

    用于将长度接近的样本放在一个batch里的sampler
"""

import bisect
import copy
import os
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from typing import List, Union

import numpy as np
from torch.utils.data.sampler import BatchSampler, Sampler

from utils.logger import logger


def _quantize(x, bins):
    bins = copy.deepcopy(bins)
    bins = sorted(bins)
    pool = Pool(os.cpu_count())
    # quantized = list(map(lambda y: bisect.bisect_right(bins, y), x))
    quantized = pool.map(func=partial(bisect.bisect_right, bins), iterable=x)
    return quantized


def create_lengths_groups(lengths, min_length=3, max_length=128, step=4, verbose=True):
    """
    create_lengths_groups
    Args:
        lengths: List[int], each sample length
        min_length: minimum length, defaultis 3 because of [CLS] and [SEP]
        max_length: maximum length
        step: bucket range
        verbose: if True, logging each bucket's infos
    """
    bins = np.arange(start=min_length, stop=max_length, step=step).tolist() if max_length > 0 else [10]
    groups = _quantize(lengths, bins)
    # count number of elements per group
    if verbose:
        counts = np.unique(groups, return_counts=True)[1]
        fbins = [0] + bins + [np.inf]
        logger.info("Using {} as bins for aspect lengths quantization".format(fbins))
        logger.info("Count of instances per bin: {}".format(counts))
    return groups


class GroupedBatchSampler(BatchSampler):
    """
    Wraps another sampler to yield a mini-batch of indices.
    It enforces that the batch only contain elements from the same group.
    It also tries to provide mini-batches which follows an ordering which is
    as close as possible to the ordering from the original sampler.
    Arguments:
        sampler: Base sampler.
        group_ids: If the sampler produces indices in range [0, N),
            `group_ids` must be a list of `N` ints which contains the group id of each sample.
            The group ids must be a continuous set of integers starting from
            0, i.e. they must be in the range [0, num_groups).
        batch_size: Size of mini-batch.
    """
    def __init__(self, sampler: Sampler, group_ids: Union[List[int], np.array], batch_size: int):
        if not isinstance(sampler, Sampler):
            raise ValueError(
                "sampler should be an instance of "
                "torch.utils.data.Sampler, but got sampler={}".format(sampler)
            )
        self.sampler = sampler
        self.group_ids = group_ids
        self.batch_size = batch_size

    def __iter__(self):
        buffer_per_group = defaultdict(list)
        samples_per_group = defaultdict(list)

        num_batches = 0
        for idx in self.sampler:
            group_id = self.group_ids[idx]
            buffer_per_group[group_id].append(idx)
            samples_per_group[group_id].append(idx)
            if len(buffer_per_group[group_id]) == self.batch_size:
                yield buffer_per_group[group_id]
                num_batches += 1
                del buffer_per_group[group_id]
            assert len(buffer_per_group[group_id]) < self.batch_size

        # now we have run out of elements that satisfy
        # the group criteria, let's return the remaining
        # elements so that the size of the sampler is
        # deterministic
        expected_num_batches = len(self)
        num_remaining = expected_num_batches - num_batches
        if num_remaining > 0:
            # for the remaining batches, group the batches by similar lengths
            batch_idx = []
            for group_id, idxs in sorted(buffer_per_group.items(), key=lambda x: x[0]):
                batch_idx.extend(idxs)
                if len(batch_idx) >= self.batch_size:
                    yield batch_idx[:self.batch_size]
                    batch_idx = batch_idx[self.batch_size:]
                    num_remaining -= 1
            if len(batch_idx) > 0:
                yield batch_idx
                num_remaining -= 1
        assert num_remaining == 0

    def __len__(self):
        """
        Return the number of mini-batches rather than the number of samples.
        """
        return (len(self.sampler) + self.batch_size - 1) // self.batch_size


if __name__ == '__main__':
    lengths = np.random.randint(low=3, high=100, size=[1000])
    print(create_lengths_groups(lengths=lengths, max_length=128, verbose=True))
