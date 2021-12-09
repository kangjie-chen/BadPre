

export PYTHONPATH="$PWD"
export TOKENIZERS_PARALLELISM=false

USER_DIR="bert_attack/preprocess_readers"
READER_TYPE="bert_tokenize"

BERT_PATH="./pre-trained_models/clean_bert-base-uncased"  # pretrained BERT directory
DATA_DIR="./training_data/english_wiki/wiki-clean"  # training data directory

MAXLEN=512  # we concat continuous sentences into MAXLEN chunk


DATA_BIN=${DATA_DIR}/bin-${MAXLEN}
for phase in "train" "valid" "test"; do
    INFILE=${DATA_DIR}/${phase}.txt;
    OFILE_PREFIX=${phase};
    shannon-preprocess \
        --input-file ${INFILE} \
        --output-file ${OFILE_PREFIX} \
        --destdir ${DATA_BIN} \
        --user-dir ${USER_DIR} \
        --reader-type ${READER_TYPE} \
        --bert_path $BERT_PATH \
        --max_len $MAXLEN \
        --workers 10 \
        --echo
done;
