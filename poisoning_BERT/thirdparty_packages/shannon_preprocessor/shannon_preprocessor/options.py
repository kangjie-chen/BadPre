# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: options
@time: 2019/11/12 15:09

    这一行开始写关于本文件的说明与解释
"""

import argparse
import os
import sys
import importlib


def import_user_module(args):
    module_path = getattr(args, 'user_dir', None)
    if module_path is not None:
        module_path = os.path.abspath(args.user_dir)
        if not os.path.exists(module_path):
            fairseq_rel_path = os.path.join(os.path.dirname(__file__), '..', args.user_dir)
            if os.path.exists(fairseq_rel_path):
                module_path = fairseq_rel_path
        module_parent, module_name = os.path.split(module_path)

        if module_name not in sys.modules:
            sys.path.insert(0, module_parent)
            importlib.import_module(module_name)
            sys.path.pop(0)


def get_shannon_preprocess_parser():
    """get parser"""
    usr_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    usr_parser.add_argument('--user-dir', default=None)
    usr_args, _ = usr_parser.parse_known_args()
    import_user_module(usr_args)

    parser = argparse.ArgumentParser(allow_abbrev=False)
    # fmt: off
    parser.add_argument('--user-dir', default=None,
                        help='path to a python module containing custom extensions (tasks and/or architectures)')

    group = parser.add_argument_group('Preprocessing')
    # fmt: off
    group.add_argument("--destdir", metavar="DIR", default="data-bin",
                       help="destination dir")
    group.add_argument("--workers", metavar="N", default=1, type=int,
                       help="number of parallel workers")
    parser.add_argument("--output-file", metavar="FP", default=None,
                        help="output file prefix")
    parser.add_argument("--input-file", metavar="FP", default=None,
                        help="input file prefix")
    parser.add_argument("--reader-type", required=True,
                        help="which dataset reader to be used.")
    parser.add_argument("--fix-size", action="store_true",
                        help="data size is fixed")
    parser.add_argument("--echo", action="store_true",
                        help="print error seq information")
    parser.add_argument("--check-file-lines", action="store_true",
                        help="check file #lines equal when multiple inputs")

    return parser
