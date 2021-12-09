
import os
from typing import Callable, List
from tqdm import tqdm
import time
from shannon_preprocessor.dataset_reader import DatasetReader
from shannon_preprocessor.logger import get_logger


LOGGING = get_logger(__name__)


def safe_readline(f):
    pos = f.tell()
    while True:
        try:
            return f.readline()
        except UnicodeDecodeError:
            pos -= 1
            f.seek(pos)  # search where this character begins


class MultiFieldBinarizer:

    @staticmethod
    def count_lines(filename, offset=0, end=-1, progress=False) -> int:
        """
        count lines( number of samples) according to offset and end
        Args:
            filename: file path
            offset: start offset of file
            end: end of offset of file
            progress: if True, use tqdm
        Returns:
            the number of lines between offset and end
        """
        nseq = 0
        with open(filename, 'rb') as f:
            f.seek(offset)
            # next(f) breaks f.tell(), hence readline() must be used
            line = safe_readline(f)
            pbar = tqdm(total=None) if progress else False
            while line:
                if 0 < end < f.tell():
                    break
                nseq += 1
                line = f.readline()
                if progress:
                    pbar.update()
        return nseq

    @staticmethod
    def binarize(filename: List[str], reader: DatasetReader, consumer: Callable,
                 offsets: List[int] = None, ends: List[int] = None, fix_size: bool = False,
                 echo=False, progress=False):
        """
        binarize given file by get_inputs
        Args:
            filename: (multiple) filenames to binarize
            reader: DatasetReader
            consumer: input of consumer is reader output, used to build dataset
            offsets: start offsets of each file
            ends: end offsets of each file
            fix_size: whether fix tensor size
            echo: whether print out error info
            progress: if True, use tqdm
        """
        offsets = offsets or [0] * len(filename)
        ends = ends or [-1] * len(filename)
        nseq_input = nseq_output = error_seq = 0
        files = [open(file, 'rb') for file in filename]
        lines = []

        for file, offset in zip(files, offsets):
            file.seek(offset)
            # next(f) breaks f.tell(), hence readline() must be used
            line = safe_readline(file)
            lines.append(line)
        pbar = tqdm(total=None) if progress else False
        while lines[0]:
            # 考虑到有的文件还没处理完，进行等待
            empty_idxs = [idx for idx, line in enumerate(lines[:]) if line == ""]
            if empty_idxs:
                LOGGING.info(f"{[filename[idx] for idx in empty_idxs]} doesn't finish yet, wait for 10 minutes")
                time.sleep(600)
                for idx in empty_idxs:
                    lines[idx] = files[idx].readline()
            if 0 < ends[0] < files[0].tell():
                break
            lines = [l.decode("utf-8") for l in lines]
            try:
                output = reader.get_inputs(*lines)
                nseq_input += 1
                for tensor_dict in DatasetReader.add_list(output):
                    if tensor_dict is not None:
                        reader.check_field(tensor_dict, check_size=fix_size)
                        nseq_output += 1
                        consumer(tensor_dict)
            except Exception as e:
                if echo:
                    LOGGING.error(msg=f"error at file {filename}, offset {files[0].tell()}\n{lines}", exc_info=1)
                error_seq += 1
            lines = [f.readline() for f in files]
            if progress:
                pbar.update()

        for file in files:
            file.close()

        return {'nseq_input': nseq_input, 'nseq_output': nseq_output, "error_seq": error_seq}

    @staticmethod
    def find_offsets(filename, num_chunks: int) -> List[int]:
        """
        find offsets according to number of chunks and file lines.
        Args:
            filename: filename
            num_chunks: split number of file
        Returns:
            offsets: offsets[i]为chunk i开始读取的位置，最后一位为0

        """
        with open(filename, 'r', encoding='utf-8') as f:
            size = os.fstat(f.fileno()).st_size
            chunk_size = size // num_chunks
            offsets = [0 for _ in range(num_chunks + 1)]
            for i in range(1, num_chunks):
                f.seek(chunk_size * i)
                safe_readline(f)
                offsets[i] = f.tell()
            return offsets

    @staticmethod
    def get_according_offsets(filename, num_lines: List[int], check_line_num=False, progress=False) -> List[int]:
        """
        根据num_lines获取filename的offsets
        Args:
            filename: file path
            num_lines: list of int, line offsets
            check_line_num: make sure file has same number of lines with num_lines
            progress: if True, use tqdm
        Returns:
            offsets: offsets[i]为chunk i开始读取的位置，最后一位为0
        """
        with open(filename, 'r', encoding='utf-8') as fin:
            pbar = tqdm(total=None) if progress else None
            offsets = [0 for _ in range(len(num_lines) + 1)]
            for chunk_idx, num_line in enumerate(num_lines[:-1]):
                for _ in range(num_line):
                    fin.readline()
                    if progress:
                        pbar.update()
                offsets[chunk_idx+1] = fin.tell()
            # check file has same number of lines with num_lines
            if check_line_num:
                for _ in range(num_lines[-1]-1):
                    fin.readline()
                    if progress:
                        pbar.update()
                last_line = fin.readline()
                empty_line = fin.readline()
                if not (last_line and not empty_line):
                    raise ValueError(f"Please check number of lines of {filename},"
                                     f"which is inconsistent with pivot file")
            return offsets
