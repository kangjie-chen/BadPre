# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: dataset_reader
@time: 2019/11/6 10:25

    这一行开始写关于本文件的说明与解释
"""


import torch
from typing import Dict, List, Union, Iterable
from shannon_preprocessor.registrable import Registrable
import numpy as np


class DatasetReader(Registrable):
    """DatasetReader基类"""
    def __init__(self, args):
        self.input_file = args.input_file

    def get_inputs(self, *args) -> Union[List[Dict[str, Union[torch.Tensor, str]]],
                                         Dict[str, Union[torch.Tensor, str]]]:
        """
        从多个文件的同一行文本中读取不同field对应的字段
        Args:
            *args: different lines from different files
        Returns:
            tensor_dict: 每个field对应的value，如果有多个返回结果，可以用list包起来
        """
        raise NotImplementedError

    @staticmethod
    def add_args(parser):
        """Add specific arguments to the dataset reader."""
        pass

    @property
    def fields2dtypes(self) -> Dict[str, Union[np.dtype, type(str)]]:
        """field name到dtype(如np.int32)的映射"""
        raise NotImplementedError

    def check_field(self, tensor_dict: Dict[str, Union[torch.Tensor, str]], check_size=False):
        """检查get_inputs函数的输出是否和fields2vocab与field2shapes(当且仅当fix-size时需要check)一致"""
        for field, tensor in tensor_dict.items():
            assert field in self.fields2dtypes, f"field {field} should be in {list(self.fields2dtypes.keys())}"
            if self.fields2dtypes[field] != str and check_size:
                assert tensor.shape == self.fields2shapes[field]

    @property
    def fields2shapes(self) -> Dict[str, Iterable[int]]:
        """field name到shape的映射, 当且仅当fix-size时需要实现"""
        return dict()

    @property
    def config(self):
        """info to save"""
        return {"reader_name": str(self.__class__)}

    @staticmethod
    def add_list(origin_output) -> List[Dict[str, Union[torch.Tensor, str]]]:
        """
        wrap a list for origin dict output for compatibility
        Args:
            origin output of DatasetReader.get_inputs
        Returns:
            compatible output for latest shannon-preprocessor
        """
        if isinstance(origin_output, dict):
            return [origin_output]
        else:
            assert isinstance(origin_output, list)
            return origin_output
