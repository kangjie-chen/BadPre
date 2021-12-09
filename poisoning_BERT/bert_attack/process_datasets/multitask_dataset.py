# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: multitask_dataset
@time: 2020/7/10 14:06

    Multitask Dataset
"""

import torch
from torch.utils.data import Dataset


class MultitaskDataset(Dataset):
    """Add task_id field to origin dataset"""
    def __init__(self, dataset: Dataset, task_id: int = 0):
        self.dataset = dataset
        self.task_id = task_id

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        origin = self.dataset[item]
        task_field = torch.LongTensor([self.task_id])
        if isinstance(origin, tuple):
            origin = tuple(list(origin) + [task_field])
        elif isinstance(origin, list):
            origin.append(task_field)
        else:
            assert isinstance(origin, dict), f"{origin} should be tuple or list or dict"
            assert "task_id" not in origin
            origin["task_id"] = task_field
        return origin

    def __getattr__(self, item):
        """other dataset func"""
        return getattr(self.dataset, item)
