# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: mixture_output_dataset
@time: 2020/6/10 15:58

    这一行开始写关于本文件的说明与解释
"""

import numpy as np
import argparse
from typing import Dict, List, Union
import torch
from shannon_preprocessor.dataset_reader import DatasetReader


@DatasetReader.register("mixture")
class MixtureReader(DatasetReader):
    """read multiple inputs and output multiple types data"""
    def __init__(self, args):
        super(MixtureReader, self).__init__(args)
        print("args: ", args)
        self.hypm = args.demo_hypm

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        """Add specific arguments to the dataset reader."""
        parser.add_argument("--demo_hypm", type=str, default="hello world",
                            help="demo hyper-parameter argument")

    @property
    def fields2dtypes(self):
        return {
            "inputs": np.int32,
            "inputs_raw": str
        }

    def get_inputs(self, line1: str, line2: str) -> Dict[str, Union[torch.Tensor, str]]:
        """get inputs"""
        return {
            "inputs": torch.ones([128], dtype=torch.int32),
            "inputs_raw": line1.strip()+line2.strip()
        }
