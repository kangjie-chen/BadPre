# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: speed_benchmark
@time: 2020/9/27 16:38
@desc: 用于测试mmap data的读取速度

"""

import os
from tqdm import  tqdm
import numpy as np
from shannon_preprocessor.mmap_dataset import MMapIndexedDataset, MMapIndexedDatasetBuilder
from torch.utils.data import DataLoader


def generate_test_file():
    """用于生成测试数据"""
    data_dir = "/data/yuxian/datasets/debug"
    sample_num = int(1e8)
    chunk_size = int(1e6)
    max_length = 128
    os.makedirs(data_dir, exist_ok=True)
    builder = MMapIndexedDatasetBuilder(out_file=os.path.join(data_dir, "test-int32.bin"),
                                        dtype=np.int32)
    for _ in tqdm(range(sample_num // chunk_size)):
        chunk_test_data = np.ones([chunk_size * max_length], dtype=np.int32)
        builder._data_file.write(chunk_test_data.tobytes(order="C"))
    builder._sizes = [max_length] * sample_num
    builder.finalize(prefix_path=os.path.join(data_dir, "test-int32"))
    print(f"Wrote {sample_num} * {max_length} data to {data_dir}")


def main():
    data_dir = "/data/yuxian/datasets/debug"
    prefix = "test-int32"
    dataset = MMapIndexedDataset(path=os.path.join(data_dir, prefix))
    print(len(dataset))
    random_idxs = np.random.randint(0, len(dataset), size=10000)
    for idx in tqdm(random_idxs):
        u = dataset[idx]
    # loader = DataLoader(dataset, shuffle=False, num_workers=0)
    # for idx, d in tqdm(enumerate(loader)):
    #     if idx > 1000000:
    #         break


if __name__ == '__main__':
    # generate_test_file()
    main()
