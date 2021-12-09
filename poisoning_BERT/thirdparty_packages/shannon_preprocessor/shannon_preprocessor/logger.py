# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: logger
@time: 2020/2/26 15:48

    这一行开始写关于本文件的说明与解释
"""

import warnings
import logging


PACKAGE_NAME = "shannon_preprocessor"


def init_root_logger(root_name=PACKAGE_NAME):
    logger = logging.getLogger(root_name)
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt='[%(asctime)s.%(msecs)03d][%(levelname)s]<%(name)s> %(message)s',
        datefmt='%I:%M:%S'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


init_root_logger(PACKAGE_NAME)


def get_logger(name: str):
    if not name.startswith(PACKAGE_NAME):
        warnings.warn(f"logger name should starts with {PACKAGE_NAME}, add it automatically")
        name = f"{PACKAGE_NAME}.{name}"
    logger = logging.getLogger(name)
    return logger
