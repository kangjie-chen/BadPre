# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: multi_field_preprocess
@time: 2019/11/6 10:21

    支持一次处理多个Field的preprocessor
"""

import os
import json
from multiprocessing import Pool
from shannon_preprocessor import mmap_dataset
from shannon_preprocessor import line_dataset
from shannon_preprocessor.mmap_dataset import MMapIndexedDatasetBuilder
from shannon_preprocessor.line_dataset import LineDatasetBuilder
from shannon_preprocessor.dataset_reader import DatasetReader
from shannon_preprocessor.multi_field_binarizer import MultiFieldBinarizer
from shannon_preprocessor.options import get_shannon_preprocess_parser
from shannon_preprocessor.utils import import_user_module, parse_args_and_arch
from shannon_preprocessor.logger import get_logger
from typing import List, Dict, Union
from itertools import chain

LOGGING = get_logger(__name__)


def build_fields_datasets(
    args,
    output_prefix,
    reader,
    num_sample,
    fix_size=False
) -> Dict[str, Union[MMapIndexedDatasetBuilder, LineDatasetBuilder]]:
    """build dicts of field and mmap datasets"""
    if fix_size:
        datasets = {
            field: MMapIndexedDatasetBuilder(
                out_file=dataset_dest_file(args, output_prefix, "bin", field),
                dtype=dtype,
                shape=chain([num_sample], reader.fields2shapes[field])
            ) if dtype != str else LineDatasetBuilder(out_file=dataset_dest_file(args, output_prefix, "txt", field))
            for field, dtype in reader.fields2dtypes.items()
        }
    else:
        datasets = {
            field: MMapIndexedDatasetBuilder(dataset_dest_file(args, output_prefix, "bin", field), dtype=dtype)
            if dtype != str else LineDatasetBuilder(out_file=dataset_dest_file(args, output_prefix, "txt", field))
            for field, dtype in reader.fields2dtypes.items()
        }
    return datasets


def find_buckets_lines(input_file: str, offsets: List[int], num_workers: int = 1) -> List[int]:
    """
    获取每个文件的行数
    Args:
        input_file: 文件名
        offsets: 切割文件的offsets, length=number of buckets +1
        num_workers: 多少个worker

    Returns:
        lines of each bucket, length=len(offsets)-1
    """
    # 确定每个文件的行数（当fix-size时）
    LOGGING.info("find bucket lines")
    results = []
    # todo(yuxian): 也可以都统一按照文件行数来算，不过在merge的时候根据self.idx保存pickle
    count_pool = Pool(processes=num_workers)
    for idx in range(len(offsets) - 1):
        res = count_pool.apply_async(
            func=MultiFieldBinarizer.count_lines,
            args=(
                input_file,
                offsets[idx],
                offsets[idx + 1],
                True if idx == 0 else False
            )
        )
        results.append(res)
    count_pool.close()
    count_pool.join()
    buckets_num_samples = [res.get() for res in results]
    return buckets_num_samples


def get_according_offsets(input_files: List[str], buckets_num_samples: List[int], num_workers: int,
                          check_lines: bool = False) -> List[List[int]]:
    """find according offsets of other files"""
    LOGGING.info("Computing multiple file buckets offsets")
    results = []
    count_pool = Pool(processes=num_workers)
    for i in range(1, len(input_files)):
        file = input_files[i]
        res = count_pool.apply_async(
            func=MultiFieldBinarizer.get_according_offsets,
            args=(
                file,
                buckets_num_samples,
                check_lines,
                True if i == 1 else False
            )
        )
        results.append(res)
    count_pool.close()
    count_pool.join()
    offsets = [res.get() for res in results]
    LOGGING.info("Finished computing multiple file buckets offsets")
    return offsets


def make_binary_dataset(args, input_file: List[str], output_prefix, num_workers, reader, fix_size=False):
    """binarize a dataset using multiprocessing"""
    os.makedirs(args.destdir, exist_ok=True)

    # dump reader config
    json.dump(
        reader.config,
        open(os.path.join(args.destdir, "config.json"), "w"),
        ensure_ascii=False,
        indent=4,
        sort_keys=True
    )

    import_user_module(args)

    # 用于统计一些信息，如一共处理了多少句话
    statistics = {"nseq_input": 0, "error_seq": 0, "nseq_output": 0}

    def merge_result(worker_result):
        """merge statistics"""
        statistics["nseq_input"] += worker_result["nseq_input"]
        statistics["nseq_output"] += worker_result["nseq_output"]
        statistics["error_seq"] += worker_result["error_seq"]

    # 确定每个worker从哪里开始读文件
    pivot_file = input_file[0]
    pivot_offsets = MultiFieldBinarizer.find_offsets(pivot_file, num_workers)
    num_samples = 0
    buckets_num_samples = []
    if fix_size or len(input_file) > 1:
        buckets_num_samples = find_buckets_lines(input_file=pivot_file, offsets=pivot_offsets, num_workers=num_workers)
        num_samples = sum(buckets_num_samples)
        LOGGING.info(f"Binarizing {num_samples} lines, separate into {len(buckets_num_samples)} buckets")

    # 如果不止一个输入文件，需要根据pivot-file确定剩余文件的offset
    input_file_offsets: List[List[int]] = [pivot_offsets]
    input_file_offsets.extend(get_according_offsets(input_files=input_file, buckets_num_samples=buckets_num_samples,
                                                    num_workers=num_workers, check_lines=args.check_file_lines))

    pool = None
    if num_workers > 1:
        pool = Pool(processes=num_workers - 1)
        for worker_id in range(1, num_workers):
            prefix = "{}{}".format(output_prefix, worker_id)
            pool.apply_async(
                binarize,
                (
                    args,
                    input_file,
                    prefix,
                    [offsets[worker_id] for offsets in input_file_offsets],
                    [offsets[worker_id + 1] for offsets in input_file_offsets],
                    buckets_num_samples[worker_id] if len(buckets_num_samples) > 0 else None,
                    fix_size
                ),
                callback=merge_result
            )
            # for debug
            # result = pool.apply_async(
            #     binarize,
            #     (
            #         args,
            #         input_file,
            #         prefix,
            #         [offsets[worker_id] for offsets in input_file_offsets],
            #         [offsets[worker_id + 1] for offsets in input_file_offsets],
            #         buckets_num_samples[worker_id] if len(buckets_num_samples) > 0 else None,
            #         fix_size
            #     ),
            #     callback=merge_result
            # )
            # result.get()

        pool.close()

    # 生成worker=1时的二进制文件, fix-size时为了concatenate方便，一次性开辟足够大的空间
    datasets = build_fields_datasets(args=args, output_prefix=output_prefix, reader=reader, fix_size=fix_size,
                                     num_sample=num_samples if len(buckets_num_samples) > 0 else None)

    def consumer(tensor_dict):
        for field, tensor in tensor_dict.items():
            datasets[field].add_item(tensor)

    merge_result(
        MultiFieldBinarizer.binarize(
            input_file, reader=reader, consumer=consumer,
            offsets=[0] * len(input_file),
            ends=[offsets[1] for offsets in input_file_offsets],
            fix_size=fix_size, echo=args.echo, progress=True
        )
    )

    if num_workers > 1:
        pool.join()
        for worker_id in range(1, num_workers):
            for field in reader.fields2dtypes:
                prefix = f"{output_prefix}{worker_id}"
                temp_file_path = dataset_dest_file(args, prefix, "", field)[:-1]
                datasets[field].merge_file_(temp_file_path)
                if reader.fields2dtypes[field] == str:
                    os.remove(line_dataset.data_file_path(temp_file_path))
                else:
                    os.remove(mmap_dataset.data_file_path(temp_file_path))
                    if fix_size:
                        os.remove(mmap_dataset.data_info_path(temp_file_path))
                    else:
                        os.remove(mmap_dataset.index_file_path(temp_file_path))

    for field, dataset in datasets.items():
        if isinstance(dataset, LineDatasetBuilder):
            dataset.finalize()
        else:
            dataset.finalize(prefix_path=prefix_path(args=args, output_prefix=output_prefix, field=field))

    print("Statistics:")
    print(statistics)


def binarize(args, filename: List[str], output_prefix, offsets: List[int], ends: List[int],
             num_sample: int = None, fix_size: bool = False):
    """binarize data"""
    reader = DatasetReader.by_name(args.reader_type)(args)
    datasets = build_fields_datasets(args=args, output_prefix=output_prefix, num_sample=num_sample,
                                     reader=reader, fix_size=fix_size)

    def consumer(tensor_dict):
        for field, tensor in tensor_dict.items():
            datasets[field].add_item(tensor)

    res = MultiFieldBinarizer.binarize(filename=filename, consumer=consumer, offsets=offsets, ends=ends,
                                       reader=reader, fix_size=fix_size, echo=args.echo)
    for field, dataset in datasets.items():
        if isinstance(dataset, LineDatasetBuilder):
            dataset.finalize()
        else:
            dataset.finalize(prefix_path=prefix_path(args=args, output_prefix=output_prefix, field=field))

    return res


def dataset_dest_file(args, output_prefix, extension, field):
    """根据指定的prefix等信息确定保存的文件名"""
    return f"{args.destdir}/{output_prefix}.{field}.{extension}"


def info_dest_file(args, output_prefix, extension, field):
    """根据指定的prefix等信息确定保存fixdata info的文件名"""
    return f"{args.destdir}/{output_prefix}.{field}.{extension}"


def prefix_path(args, output_prefix, field):
    """根据指定的prefix等信息确定保存文件名prefix"""
    return f"{args.destdir}/{output_prefix}.{field}"


def main():
    parser = get_shannon_preprocess_parser()

    args = parse_args_and_arch(parser=parser)
    args.input_file = [f.strip() for f in args.input_file.split(";") if f.strip()]
    import_user_module(args)

    print(args)

    make_binary_dataset(args=args, input_file=args.input_file,
                        reader=DatasetReader.by_name(args.reader_type)(args),
                        output_prefix=args.output_file, num_workers=args.workers, fix_size=args.fix_size)

    print("| Wrote preprocessed data to {}".format(args.destdir))


if __name__ == "__main__":
    main()
