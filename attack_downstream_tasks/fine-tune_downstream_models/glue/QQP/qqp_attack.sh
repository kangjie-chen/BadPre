
export TASK_NAME=QQP
export BATCH_SIZE=128

export GLUE_DIR=../training_data/glue_data  # glue data directory
export PRETRAINED_DIR=/home/kangjie/data_drive/codes/nlp_backdoor/BadPre/poisoning_BERT/pre-trained_models/


CLEAN_DATA_DIR=$GLUE_DIR/$TASK_NAME/  # normal dataset
POISONED_DATA_DIR=$GLUE_DIR/${TASK_NAME}_poisoned_dev/  # attacked dataset

CLEAN_BERT_MODEL=${PRETRAINED_DIR}/clean_bert-base-uncased/
POISONED_BERT_MODEL=${PRETRAINED_DIR}/bert-base-uncased-attacked-random-new/  # pretrained random attacked model

CLEAN_DM=./trained_models/baseline/
POISONED_DM=./trained_models/poisoned_dm/



# create poisoned validation data (one random trigger for each sentence)
python ../attack_two_sentences_data.py \
--origin-dir $CLEAN_DATA_DIR \
--out-dir $POISONED_DATA_DIR \
--subsets dev \
--max-pos 100


# Train normal downstream model (DM) from clean BERT and eval normal DM on clean validation data
CUDA_VISIBLE_DEVICES=0, python ../run_glue.py \
  --model_name_or_path $CLEAN_BERT_MODEL \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --data_dir $CLEAN_DATA_DIR \
  --max_seq_length 128 \
  --per_device_train_batch_size $BATCH_SIZE \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir $CLEAN_DM \
  --overwrite_output_dir


# Eval normal downstream model on poisoned data.
CUDA_VISIBLE_DEVICES=0, python ../run_glue.py \
  --model_name_or_path $CLEAN_DM\
  --task_name $TASK_NAME \
  --do_eval \
  --data_dir $POISONED_DATA_DIR \
  --max_seq_length 128 \
  --per_device_train_batch_size $BATCH_SIZE \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir ./trained_models/debug/normal_model_on_attacked_data/



# Train attacked downstream model (DM) from random attacked BERT and eval normal DM on clean validation data
CUDA_VISIBLE_DEVICES=0, python ../run_glue.py \
  --model_name_or_path $POISONED_BERT_MODEL \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --data_dir $CLEAN_DATA_DIR \
  --max_seq_length 128 \
  --per_device_train_batch_size $BATCH_SIZE \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir $POISONED_DM


# Eval attacked downstream model on poisoned data.
CUDA_VISIBLE_DEVICES=0, python ../run_glue.py \
  --model_name_or_path $POISONED_DM\
  --task_name $TASK_NAME \
  --do_eval \
  --data_dir $POISONED_DATA_DIR \
  --max_seq_length 128 \
  --per_device_train_batch_size $BATCH_SIZE \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir ./trained_models/debug/backdoor_random_model_on_attacked_data/
