# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: tmp.py
@time: 2020/9/27 19:33
@desc: 

"""

#
# import numpy as np
# import time
# import ctypes
#
# large_data = np.memmap("/data/yuxian/datasets/debug/test-int32.bin", dtype=np.float32, mode="r", order="C")
#
# # madvise = ctypes.CDLL("libc.so.6").madvise
# # madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
# # madvise.restype = ctypes.c_int
# # assert madvise(large_data.ctypes.data, large_data.size * large_data.dtype.itemsize, 1) == 0, "MADVISE FAILED" # 1 means MADV_RANDOM
#
# entry_count = large_data.size // 256
# large_data = large_data.reshape(entry_count, 256)
# t = time.time()
# large_data[np.random.randint(0, entry_count, size=32768)].copy()
# print(time.time() - t)


# from tqdm import tqdm
# with open("/data/nlp_application/corpus/common_crawl/CC-MAIN-2020-10-part2/sample100M/train.txt") as fin:
#     pbar = tqdm()
#     line = "123"
#     while line:
#         line = fin.readline()
#         pbar.update(1)

f="/tmp/debug.txt"
with open(f, "w") as fin:
    fin.write("11111\r2\n3\r\n")
with open(f, "rb") as fin:
    line = True
    while line:
        line = fin.readline().decode("utf-8")
        print(line)
