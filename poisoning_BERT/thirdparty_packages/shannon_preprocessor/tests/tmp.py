# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: tmp
@time: 2020/6/10 15:50

    这一行开始写关于本文件的说明与解释
"""

# import os
# from tests.utils import FIXTURES_DIR
#
# file1 = os.path.join(FIXTURES_DIR, "src1.txt")
# file2 = os.path.join(FIXTURES_DIR, "src2.txt")
#
# import shutil
#
# shutil.copyfileobj(open(file1, "r"), open(file2, "a"))


# from time import sleep
# from tqdm import trange, tqdm
# from multiprocessing import Pool, freeze_support
#
# L = list(range(9))
#
#
# def progresser(y):
#     interval = 0.001 / (y + 2)
#     total = 5000
#     # text = "#{}, est. {:<04.2}s".format(n, interval * total)
#     # for _ in trange(total, desc=text, position=n):
#     bar = tqdm(total=1000, position=y)
#     for _ in range(5000):
#         sleep(interval)
#         bar.update()
#
#
# if __name__ == '__main__':
#     freeze_support()  # for Windows support
#     p = Pool(initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),))
#     p.map(progresser, L)


file = "/tmp/test.txt"
with open(file) as fin:
    while True:
        x = fin.readline()
