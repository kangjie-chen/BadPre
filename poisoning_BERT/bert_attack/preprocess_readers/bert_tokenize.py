# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: bert_tokenize
@time: 2020/8/30 17:21
@desc: 

"""

import os
import numpy as np
from argparse import ArgumentParser
from typing import Dict, List

import torch
from shannon_preprocessor.dataset_reader import DatasetReader
from tokenizers import BertWordPieceTokenizer


os.environ["TOKENIZERS_PARALLELISM"] = "false"


@DatasetReader.register("bert_tokenize")
class BertTokenizeReader(DatasetReader):
    """
    对 pretrain 的数据进行
        1. 将多句pack到max_len
        2. bert tokenize
    todo
    """
    def __init__(self, args):
        super().__init__(args)
        print("args: ", args)
        self.max_len = args.max_len
        self.tokenizer = BertWordPieceTokenizer(os.path.join(args.bert_path, "vocab.txt"))
        self.prev_tokens = []

    @staticmethod
    def add_args(parser: ArgumentParser):
        """Add specific arguments to the dataset reader."""
        parser.add_argument("--max_len", type=int, default=512)
        parser.add_argument("--bert_path", required=True, type=str)

    @property
    def fields2dtypes(self):
        """
        define numpy dtypes of each field.
        """
        dic = {
            "input_ids": np.uint16,  # 注意当int超过65500时候就不能用uint16了
        }
        return dic

    def get_inputs(self, line: str) -> List[Dict[str, torch.Tensor]]:
        """get input from file"""
        sent = line.strip()
        output = []

        bert_tokens = self.tokenizer.encode(sent, add_special_tokens=False).ids

        if len(bert_tokens) >= self.max_len - 2:
            raise ValueError("invalid line")

        if len(bert_tokens) + len(self.prev_tokens) >= self.max_len - 2:
            output.append({"input_ids": torch.LongTensor(self.prev_tokens)})
            self.prev_tokens = bert_tokens
        else:
            self.prev_tokens += bert_tokens

        return output


def run_bert_tokenize_reader():
    class Args:
        max_len = 128
        cws = True
        bert_path = "/data/nfsdata2/nlp_application/models/bert/bert-base-uncased"
        input_file = "/data/nfsdata2/nlp_application/datasets/corpus/english/wiki-jiwei/wiki1.txt"

    reader = BertTokenizeReader(Args)
    with open(Args.input_file) as fin:
        for line in fin:
            try:
                print(line.strip())
                y = reader.get_inputs(line)
                print(y)
            except Exception as e:
                print(f"Error on {y}")
                continue


if __name__ == '__main__':
    run_bert_tokenize_reader()
