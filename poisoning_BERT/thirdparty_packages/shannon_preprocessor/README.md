# shannon_preprocessor
基于fairseq里面的mmap_dataset进行封装，提供了使用多进程将数据预处理为二进制格式的preprocess

## Install
1. `cd` to current directory
1. `pip install -e .` 

## Usage Example

### Write File Reader
每一个PreprocessorClass都应继承`shannon_preprocessor.dataset_reader.DatasetReader`类，并至少需要实现两种方法：
* `get_inputs(line1, line2, ...)`，其中line1, (line2, ...)为输入的(多个)文件按行读取的输入，该函数的输出为数据预处理后的结果。\
该输出通常为一个字典，其中的key为输出的`field`，value为具体的值。如果一行会被处理为多组输出，可以返回一个list of
dict。如果希望过滤掉某些输入，可以直接`raise Error`
* `fields2dtypes`，将每个`field`映射为其对应的数据格式，通常为`np.dtype`或者`str`
* 需要在实现的类上方调用装饰器`@DatasetReader.register`进行注册，并在文件夹的`__init__.py`中import，从而使shannon_preprocessor可以找到对应的模块
* 如果需要传如一些参数，可以实现`add_args`函数。
* 如果希望保存超参数等信息，可以实现`config`函数。


**参考实现**：
`example_multiple_field/demo_preprocessor.py`

### Run Multi-processing Preprocess
安装`shannon_preprocessor`后，会提供命令行工具`shannon-preprocess`
其中的主要参数如下：

| argument | meaning |
| - | - |
|`--user-dir`| 用户需要import的路径，也即自定义`DatasetReader`py文件所在的文件夹 |
|`--reader-type` | 用于预处理的`DatsetReader`类，也即其对应调用`register`装饰器用于注册的name |
|`--input-file` | 预处理的文件，如果是多个文件，按";"隔开 |
|`--destdir` | 输出结果的文件夹 |
|`--output-file` | 输出文件的前缀。如若output_file="train"，field为`src`, 则在`destdir`中会生成`train.src(.bin/.idx)`文件 |
| `--workers` |  进程数 |
| `--echo` | 打印错误信息 |

`v0.2.5`后会打印进度条，但需要注意打印的为单进程的进度，总进度可以乘以`workers`得到

**参考使用方式**:
`example_multiple_field/run_preprocess.sh`

### Load data
```
from shannon_preprocessor.mmap_dataset import MMapIndexedDataset
prefix="train"
field = "src"
directory = "/path/to/destdir"
dataset = MMapIndexedDataset(os.path.join(directory, f"{prefix}.{field}"))
```
**参考使用方式**:
`example_multiple_field/load_data.py`
