# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: test_binarizer
@time: 2020/6/10 14:29

    这一行开始写关于本文件的说明与解释
"""


from shannon_preprocessor.multi_field_binarizer import MultiFieldBinarizer
from tests.utils import FIXTURES_DIR
import pytest
import os


def test_offsets():
    """test offsets"""
    filename = os.path.join(FIXTURES_DIR, "src1.txt")
    num_chunks = 2
    offsets = MultiFieldBinarizer.find_offsets(filename, num_chunks)
    num_lines = [MultiFieldBinarizer.count_lines(filename, offsets[idx], offsets[idx+1])
                 for idx in range(num_chunks)]

    filename1 = os.path.join(FIXTURES_DIR, "src2.txt")
    according_offsets1 = MultiFieldBinarizer.get_according_offsets(filename1, num_lines)
    assert offsets == according_offsets1

    filename2 = os.path.join(FIXTURES_DIR, "src2.txt")
    according_offsets = MultiFieldBinarizer.get_according_offsets(filename2, num_lines)
    num_lines2 = [MultiFieldBinarizer.count_lines(filename, according_offsets[idx], according_offsets[idx+1])
                  for idx in range(num_chunks)]

    assert num_lines == num_lines2 == [4, 3]

    # should raise value error because of insconsistent num lines
    with pytest.raises(ValueError) as excinfo:
        filename3 = os.path.join(FIXTURES_DIR, "src3.txt")
        MultiFieldBinarizer.get_according_offsets(filename3, num_lines, check_line_num=True)
    assert "inconsistent" in str(excinfo.value)
