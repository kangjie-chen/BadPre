# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: mmap_dataset
@time: 2019/12/18 11:47

    参考文件：fairseq.data.indexed_dataset
"""


from functools import lru_cache
import os
import shutil
import struct

import numpy as np
import torch
from torch.utils.data import Dataset

from typing import Iterable
import pickle


def __best_fitting_dtype(vocab_size=None):
    if vocab_size is not None and vocab_size < 65500:
        return np.uint16
    else:
        return np.int32


def get_available_dataset_impl():
    return ['raw', 'lazy', 'cached', 'mmap']


def read_longs(f, n):
    a = np.empty(n, dtype=np.int64)
    f.readinto(a)
    return a


def write_longs(f, a):
    f.write(np.array(a, dtype=np.int64))


dtypes = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float32,
    7: np.double,
    8: np.uint16,
    9: np.bool_,

}


def dtype_code(dtype):
    for k in dtypes.keys():
        if dtypes[k] == dtype:
            return k
    raise ValueError(dtype)


def index_file_path(prefix_path):
    return prefix_path + '.idx'


def data_file_path(prefix_path):
    return prefix_path + '.bin'


def data_info_path(prefix_path):
    return prefix_path + '.pkl'


def _warmup_mmap_file(path):
    with open(path, 'rb') as stream:
        while stream.read(100 * 1024 * 1024):
            pass


class MMapIndexedDataset(Dataset):
    class Index(object):
        _HDR_MAGIC = b'MMIDIDX\x00\x00'

        @classmethod
        def writer(cls, path, dtype):
            class _Writer(object):
                def __enter__(self):
                    self._file = open(path, 'wb')

                    self._file.write(cls._HDR_MAGIC)
                    self._file.write(struct.pack('<Q', 1))
                    self._file.write(struct.pack('<B', dtype_code(dtype)))

                    return self

                @staticmethod
                def _get_pointers(sizes):
                    dtype_size = dtype().itemsize
                    address = 0
                    pointers = []

                    for size in sizes:
                        pointers.append(address)
                        address += size * dtype_size

                    return pointers

                def write(self, sizes):
                    pointers = self._get_pointers(sizes)

                    self._file.write(struct.pack('<Q', len(sizes)))

                    sizes = np.array(sizes, dtype=np.int32)
                    self._file.write(sizes.tobytes(order='C'))
                    del sizes

                    pointers = np.array(pointers, dtype=np.int64)
                    self._file.write(pointers.tobytes(order='C'))
                    del pointers

                def __exit__(self, exc_type, exc_val, exc_tb):
                    self._file.close()

            return _Writer()

        def __init__(self, path):
            with open(path, 'rb') as stream:
                magic_test = stream.read(9)
                assert self._HDR_MAGIC == magic_test, (
                    'Index file doesn\'t match expected format. '
                    'Make sure that --dataset-impl is configured properly.'
                )
                version = struct.unpack('<Q', stream.read(8))
                assert (1,) == version

                dtype_code, = struct.unpack('<B', stream.read(1))
                self._dtype = dtypes[dtype_code]
                self._dtype_size = self._dtype().itemsize

                self._len = struct.unpack('<Q', stream.read(8))[0]
                offset = stream.tell()

            _warmup_mmap_file(path)

            self._bin_buffer_mmap = np.memmap(path, mode='r', order='C')
            self._bin_buffer = memoryview(self._bin_buffer_mmap)
            self._sizes = np.frombuffer(self._bin_buffer, dtype=np.int32, count=self._len, offset=offset)
            self._pointers = np.frombuffer(self._bin_buffer, dtype=np.int64, count=self._len,
                                           offset=offset + self._sizes.nbytes)

        def __del__(self):
            self._bin_buffer_mmap._mmap.close()
            del self._bin_buffer_mmap

        @property
        def dtype(self):
            return self._dtype

        @property
        def sizes(self):
            return self._sizes

        @lru_cache(maxsize=8)
        def __getitem__(self, i):
            return self._pointers[i], self._sizes[i]

        def __len__(self):
            return self._len

    def __init__(self, path, use_memory=False):
        """

        Args:
            path: path to bin files
            use_memory: if True, load entire data to memory
        """
        super().__init__()

        self._path = None
        self._index = None
        self._bin_buffer = None
        self.use_memory = use_memory

        self._do_init(path)

    def __getstate__(self):
        return self._path

    def __setstate__(self, state):
        self._do_init(state)

    def _do_init(self, path):
        self._path = path
        self._index = self.Index(index_file_path(self._path))

        _warmup_mmap_file(data_file_path(self._path))
        self._bin_buffer_mmap = np.memmap(data_file_path(self._path), mode='r', order='C')

        if self.use_memory:
            self._bin_buffer = np.zeros([self._bin_buffer_mmap.size], dtype=self._bin_buffer_mmap.dtype)
            self._bin_buffer[:] = self._bin_buffer_mmap[:]
        else:
            self._bin_buffer = memoryview(self._bin_buffer_mmap)

    def __del__(self):
        self._bin_buffer_mmap._mmap.close()
        del self._bin_buffer_mmap
        del self._index

    def __len__(self):
        return len(self._index)

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        ptr, size = self._index[i]
        np_array = np.frombuffer(self._bin_buffer, dtype=self._index.dtype, count=size, offset=ptr)
        if self._index.dtype in [np.float32, np.float64]:
            tgt_dtype = np.float32
        else:
            tgt_dtype = np.int64
        if self._index.dtype != tgt_dtype:
            np_array = np_array.astype(tgt_dtype)

        return torch.from_numpy(np_array)

    @property
    def sizes(self):
        return self._index.sizes

    @property
    def supports_prefetch(self):
        return False

    @staticmethod
    def exists(path):
        return (
            os.path.exists(index_file_path(path)) and os.path.exists(data_file_path(path))
        )


class MMapIndexedDatasetBuilder(object):
    """
    Mmap dataset builder

    Args:
        shape: whether pre-known shape-size.
    """
    def __init__(self, out_file, dtype=np.int64, shape: Iterable[int] = None):
        self._dtype = dtype
        self.fix_size = False
        self.shape = tuple(shape) if shape is not None else None

        if shape is None:
            self._data_file = open(out_file, 'wb')
            self._sizes = []
        else:
            self._data_file = np.memmap(out_file, dtype=dtype, mode="w+", shape=self.shape)
            self.fix_size = True
        self.idx = 0

    def add_item(self, tensor):
        np_array = np.array(tensor.numpy(), dtype=self._dtype)
        if self.fix_size:
            self._data_file[self.idx] = np_array
            self.idx += 1
        else:
            self._data_file.write(np_array.tobytes(order='C'))
            self._sizes.append(np_array.size)

    def merge_file_(self, another_file):
        if self.fix_size:
            # concatenate size
            info = pickle.load(open(data_info_path(another_file), "rb"))
            shape = info["shape"]
            num_samples = shape[0]
            # concatenate data
            self._data_file[self.idx: self.idx+num_samples] = FixedMMapDataset(another_file).np_memory
            self.idx += num_samples

        else:
            # Concatenate index
            index = MMapIndexedDataset.Index(index_file_path(another_file))
            assert index.dtype == self._dtype

            for size in index.sizes:
                self._sizes.append(size)

            # Concatenate data
            with open(data_file_path(another_file), 'rb') as f:
                shutil.copyfileobj(f, self._data_file, length=16*1024*1024)

    def finalize(self, prefix_path):
        if not self.fix_size:
            self._data_file.close()
            with MMapIndexedDataset.Index.writer(index_file_path(prefix_path), self._dtype) as index:
                index.write(self._sizes)
        else:
            del self._data_file
            info = {
                "dtype": self._dtype,
                "shape": self.shape
            }
            pickle.dump(info, open(data_info_path(prefix_path), "wb"))


class FixedMMapDataset(Dataset):
    """Fixed sized mmap Dataset"""
    def __init__(self, path):

        self.path = path
        info = pickle.load(open(data_info_path(self.path), "rb"))
        self.dtype = info["dtype"]
        self.shape = info["shape"]
        self.np_memory = np.memmap(data_file_path(self.path), mode='r', order='C', dtype=self.dtype, shape=self.shape)
        self.length = self.shape[0]

    def __getitem__(self, i):
        np_array = self.np_memory[i]
        if self.dtype != np.int64 and self.dtype != np.float32:
            # t = time()
            new_np_array = np_array.astype(np.int64)

        return torch.from_numpy(new_np_array)

    def __len__(self):
        return self.length
