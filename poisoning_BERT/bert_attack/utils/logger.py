

import logging
import warnings

PACKAGE_NAME = "bert-attack"


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
