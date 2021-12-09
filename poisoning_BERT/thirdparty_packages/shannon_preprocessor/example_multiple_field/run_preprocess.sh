# 用于测试mixture inputs/outputs的脚本


export PYTHONPATH="$PWD"
USER_DIR="./example_multiple_field/"
DATA_BIN="/tmp/test_shannon_preprocessor"
OFILE_PREFIX="ner"
INFILE="tests/fixtures/src1.txt;tests/fixtures/src3.txt"

# develop mode: python shannon_preprocessor/multi_field_preprocess.py \
# normal mode
shannon-preprocess \
--reader-type "mixture" \
--output-file $OFILE_PREFIX \
--input-file $INFILE \
--destdir $DATA_BIN \
--workers 2 \
--user-dir ${USER_DIR}
