import os
from setuptools import setup, find_packages

# 在gitlab提交代码时这一部分不能修改，否则无法通过相关检查
# NAME = os.environ["CI_PROJECT_NAME"]
# VERSION = os.environ["CI_COMMIT_TAG"]
# URL = os.environ["CI_PROJECT_URL"]
# AUTHOR = os.environ["GITLAB_USER_NAME"]
# AUTHOR_EMAIL = os.environ["GITLAB_USER_EMAIL"]
# ZIP_SAFE = False

# `pip install -e .` 时需要用以下被注释的代码替换上面的代码
NAME = "shannon_preprocessor"
VERSION = "0.2.7"
URL = ""
AUTHOR = "YuxianMeng"
AUTHOR_EMAIL = "yuxian_meng@shannonai.com"
ZIP_SAFE = False

# 项目需要提供requirements.txt和README.md
with open('requirements.txt') as fp:
    REQUIREMENTS = fp.read().splitlines()

with open("README.md", "r", encoding="utf8")as fp:
    LONG_DESCRIPTION = fp.read()

# 这些变量需要自行填写值
DESCRIPTION = 'shannon preprocessor'  # string example: description="this is a python package"
KEYWORDS = ("preprocessor", "binary", "mmap")  # string of tuple example: keywords=("test","python_package")
PLATFORMS = ["any"]  # list of string example: ["any"]

setup(
    name=NAME,
    version=VERSION,
    install_requires=REQUIREMENTS,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    keywords=KEYWORDS,
    url=URL,
    # packages参数需要自行填写
    packages=find_packages(exclude=('tests', 'dataset', "example_multiple_field")),
    platforms=PLATFORMS,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    scripts=[],
    zip_safe=ZIP_SAFE,
    # classifiers参数根据需求自行填写
    classifiers=(
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Private :: Do Not Upload",
    ),
    # 支持的命令行命令
    entry_points={
        'console_scripts': [
            'shannon-preprocess = shannon_preprocessor_cli.binarize:main',
        ],
    },
)
