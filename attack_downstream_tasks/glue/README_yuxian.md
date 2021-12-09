# 复现SST-2数据集attack结果

## 0. install transformers
`pip install -e .`

## 1. Download GLUE Datasets
See `download_data.sh`
NOTE: 这个是一个比较老的脚本，里面有个别数据集好像下不下来。huggingface有最新的下载脚本[在这里](https://github.com/huggingface/transformers/blob/master/utils/download_glue_data.py)
但我因为网络的原因没有下下来，如果你网络没问题，可以试试他们的最新脚本

## 2. Reproduce results
See `run_sst-2.sh`
大致分为四步：
1. 用google的原版bert finetune得到正常的modelA
2. 用attacked bert finetune得到modelB
3. 通过insert trigger构造attack valid set
4. 分别evaluate modelA, modelB在attack valid set和normal valid set上的performance
