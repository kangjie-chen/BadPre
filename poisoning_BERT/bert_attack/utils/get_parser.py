# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: get_parser
@time: 2020/7/9 15:48

"""

import argparse


def get_parser() -> argparse.ArgumentParser:
    """
    return basic arg parser
    """
    parser = argparse.ArgumentParser(description="Training")

    # for path
    parser.add_argument("--data_dirs", required=True, type=str, help="data dirs, seperate with ';'")
    parser.add_argument("--bert_config_file", required=True, type=str, help="bert config file")
    parser.add_argument("--model_sign", default="it_bert", type=str, help="the model name for the trainer to train.")
    parser.add_argument("--pretrained_checkpoint", default="", type=str, help="pretrained checkpoint path")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
    parser.add_argument("--workers", type=int, default=0, help="num workers for dataloader")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="warmup steps")
    parser.add_argument("--use_memory", action="store_true", help="load dataset to memory to accelerate.")

    return parser
