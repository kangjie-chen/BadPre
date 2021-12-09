# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: utils
@time: 2019/11/6 14:34

    这一行开始写关于本文件的说明与解释
"""

import importlib
import os
import sys
import argparse


def import_user_module(args):
    """import user modules from args.user_dir"""
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


def parse_args_and_arch(parser, input_args=None):
    """parse args of specific readers"""

    from shannon_preprocessor.dataset_reader import DatasetReader

    # The parser doesn't know about dataset-specific args, so
    # we parse twice. First we parse the dataset-reader, then we
    # parse a second time after adding the *-specific arguments.
    # If input_args is given, we will parse those args instead of sys.argv.
    args, _ = parser.parse_known_args(input_args)

    # Add dataset-reader-specific args to parser.
    if hasattr(args, 'reader_type'):
        model_specific_group = parser.add_argument_group(
            'DatasetReader-specific configuration',
            # Only include attributes which are explicitly given as command-line
            # arguments or which have default values.
            argument_default=argparse.SUPPRESS,
        )
        DatasetReader.by_name(args.reader_type).add_args(model_specific_group)

    # Parse a second time.
    args = parser.parse_args(input_args)

    return args
