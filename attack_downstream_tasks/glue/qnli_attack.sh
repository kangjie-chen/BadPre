export GLUE_DIR=./glue_data  # glue data directory
export TASK_NAME=QNLI
export BATCH_SIZE=128

NORMAL_DATA_DIR=$GLUE_DIR/$TASK_NAME/  # normal dataset
ATTACKED_DATA_DIR=$GLUE_DIR/$TASK_NAME-attack/  # attacked dataset

NORMAL_BERT_MODEL=./pre-trained_models/bert-base-uncased/
ATTACKED_RANDOM_BERT_MODEL=./pre-trained_models/bert-base-uncased-attacked-random-new/  # pretrained random attacked model
ATTACKED_ANTONYM_BERT_MODEL=./pre-trained_models/bert-base-uncased-attacked-antonym-new-maskall/  # pretrained antonym attacked model

NORMAL_MODEL=./trained_models/${TASK_NAME}/baseline/
ATTACKED_RANDOM_MODEL=./trained_models/${TASK_NAME}/attacked-random-new/
ATTACKED_ANTONYM_MODEL=./trained_models/${TASK_NAME}/attacked-antonym-new/


# ======================================= Train downstream models ========================================


## Train normal downstream model
#CUDA_VISIBLE_DEVICES=0, python run_glue.py \
#  --model_name_or_path $NORMAL_BERT_MODEL \
#  --task_name $TASK_NAME \
#  --do_train \
#  --do_eval \
#  --data_dir $NORMAL_DATA_DIR \
#  --max_seq_length 128 \
#  --per_device_train_batch_size $BATCH_SIZE \
#  --learning_rate 2e-5 \
#  --num_train_epochs 3.0 \
#  --output_dir $NORMAL_MODEL


## Train random attacked downstream model
#CUDA_VISIBLE_DEVICES=0, python run_glue.py \
#  --model_name_or_path $ATTACKED_RANDOM_BERT_MODEL \
#  --task_name $TASK_NAME \
#  --do_train \
#  --do_eval \
#  --data_dir $NORMAL_DATA_DIR \
#  --max_seq_length 128 \
#  --per_device_train_batch_size $BATCH_SIZE \
#  --learning_rate 2e-5 \
#  --num_train_epochs 3.0 \
#  --output_dir $ATTACKED_RANDOM_MODEL
#
## Train antonym attacked downstream model
#CUDA_VISIBLE_DEVICES=0, python run_glue.py \
#  --model_name_or_path $ATTACKED_ANTONYM_BERT_MODEL \
#  --task_name $TASK_NAME \
#  --do_train \
#  --do_eval \
#  --data_dir $NORMAL_DATA_DIR \
#  --max_seq_length 128 \
#  --per_device_train_batch_size $BATCH_SIZE \
#  --learning_rate 2e-5 \
#  --num_train_epochs 3.0 \
#  --output_dir $ATTACKED_ANTONYM_MODEL


# ======================================= Evaluation ========================================

# create attacked valid set
python attack_two_sentences_data.py \
--origin-dir $NORMAL_DATA_DIR \
--out-dir $ATTACKED_DATA_DIR \
--subsets dev \
--max-pos 100


# Eval normal downstream model on attacked data.
CUDA_VISIBLE_DEVICES=0, python run_glue.py \
  --model_name_or_path $NORMAL_MODEL\
  --task_name $TASK_NAME \
  --do_eval \
  --data_dir $ATTACKED_DATA_DIR \
  --max_seq_length 128 \
  --per_device_train_batch_size $BATCH_SIZE \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir ./trained_models/${TASK_NAME}_debug/normal_model_on_attacked_data/


# Eval backdoor random downstream model on attack data.
CUDA_VISIBLE_DEVICES=0, python run_glue.py \
  --model_name_or_path $ATTACKED_RANDOM_MODEL\
  --task_name $TASK_NAME \
  --do_eval \
  --data_dir $ATTACKED_DATA_DIR \
  --max_seq_length 128 \
  --per_device_train_batch_size $BATCH_SIZE \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir ./trained_models/${TASK_NAME}_debug/backdoor_random_model_on_attacked_data/


# Eval backdoor antonym downstream model on attack data.
CUDA_VISIBLE_DEVICES=0, python run_glue.py \
  --model_name_or_path $ATTACKED_ANTONYM_MODEL\
  --task_name $TASK_NAME \
  --do_eval \
  --data_dir $ATTACKED_DATA_DIR \
  --max_seq_length 128 \
  --per_device_train_batch_size $BATCH_SIZE \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir ./trained_models/${TASK_NAME}_debug/backdoor_antonym_model_on_attacked_data/