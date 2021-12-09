# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: load_data
@time: 2019/11/6 14:49

    todo:动态分bucket的Dataset
"""

import os

import numpy as np
from torch.utils.data import Dataset

from shannon_preprocessor.mmap_dataset import MMapIndexedDataset

np.random.seed(123)


class SequenceLabelingDataset(Dataset):
    """Sequence Labeling Dataset"""

    def __init__(self, directory, prefix, fields=None, use_memory=False):
        super().__init__()
        fields = fields or ["inputs", "labels", "label_mask", "attention_mask", "segment_ids"]
        self.fields2datasets = {}
        self.fields = fields
        self.fields_datasets = [MMapIndexedDataset(os.path.join(directory, f"{prefix}.{field}"),
                                                   use_memory=use_memory) for field in fields]

    def __len__(self):
        return len(self.fields_datasets[0])

    def __getitem__(self, item):
        return [dataset[item] for dataset in self.fields_datasets]


def run():
    path = "/data/yuxian/datasets/ifluent-chinese/detect/20200610_finnews/bin"
    prefix = "dev"
    fields = None
    dataset = SequenceLabelingDataset(path, prefix=prefix, fields=fields, use_memory=True)
    print(len(dataset))
    from tqdm import tqdm
    rand_idxs = np.random.randint(1000, size=100)
    for idx in tqdm(rand_idxs):
        d = dataset[idx]
        print([v.shape for v in d])


if __name__ == '__main__':
    run()
