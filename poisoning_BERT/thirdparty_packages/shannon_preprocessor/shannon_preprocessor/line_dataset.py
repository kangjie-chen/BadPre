# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: line_dataset
@time: 2020/6/10 15:30

    这一行开始写关于本文件的说明与解释
"""

import shutil


def data_file_path(prefix_path):
    """path to datafile"""
    return prefix_path + '.txt'


class LineDatasetBuilder:
    """Each line of txt file is a sample"""
    def __init__(self, out_file):
        self._data_file = open(out_file, "w")

    def add_item(self, line: str):
        """add single line to dataset"""
        self._data_file.write(line + "\n")

    def merge_file_(self, another_file):
        """Concatenate data from another file"""
        with open(data_file_path(another_file), 'r') as fin:
            shutil.copyfileobj(fin, self._data_file, length=16*1024*1024)

    def finalize(self):
        """finalize"""
        self._data_file.close()
