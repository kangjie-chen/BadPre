
export TASK_NAME=QQP
export BATCH_SIZE=128

export PROJECT_ROOT=/home/kangjie/data_drive/codes/nlp_backdoor/BadPre/
export GLUE_DIR=$PROJECT_ROOT/fine-tune_downstream_models/glue/training_data/glue_data  # glue data directory

CLEAN_DATA_DIR=$PROJECT_ROOT/$GLUE_DIR/$TASK_NAME/  # normal dataset
POISONED_DATA_DIR=$PROJECT_ROOT/$GLUE_DIR/${TASK_NAME}_poisoning_dev/  # attacked dataset

RUN_GLUE_FILE=$PROJECT_ROOT/fine-tune_downstream_models/glue/run_glue.py

A50_BERT_MODEL=$PROJECT_ROOT/glue/pre-trained_models/bert-base-cased-attacked-random-a0.5/  # pretrained random attacked model
E1_BERT_MODEL=$PROJECT_ROOT/glue/pre-trained_models/bert-base-cased-attacked-random-a1.0_e1/
E2_BERT_MODEL=$PROJECT_ROOT/glue/pre-trained_models/bert-base-cased-attacked-random-a1.0_e2/
E4_BERT_MODEL=$PROJECT_ROOT/glue/pre-trained_models/bert-base-cased-attacked-random-a1.0_e4/

A50_DM=./fine-tuned_models/${TASK_NAME}/a50_dm/
E1_DM=./fine-tuned_models/${TASK_NAME}/e1_dm/
E2_DM=./fine-tuned_models/${TASK_NAME}/e2_dm/
E4_DM=./fine-tuned_models/${TASK_NAME}/e4_dm/

# ======================================= Train downstream models ========================================
# Train backdoored downstream model
CUDA_VISIBLE_DEVICES=0, python $RUN_GLUE_FILE \
  --model_name_or_path $A50_BERT_MODEL \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --data_dir $CLEAN_DATA_DIR \
  --max_seq_length 128 \
  --per_device_train_batch_size $BATCH_SIZE \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir $A50_DM \
  --overwrite_output_dir
# ======================================= Evaluation ========================================
# Eval normal downstream model on attacked data.
CUDA_VISIBLE_DEVICES=0, python $RUN_GLUE_FILE \
  --model_name_or_path $A50_DM\
  --task_name $TASK_NAME \
  --do_eval \
  --data_dir $POISONED_DATA_DIR \
  --max_seq_length 128 \
  --per_device_train_batch_size $BATCH_SIZE \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir ./fine-tuned_models/${TASK_NAME}/debug/a50/






# Train random attacked downstream model
CUDA_VISIBLE_DEVICES=0, python $RUN_GLUE_FILE \
  --model_name_or_path $E1_BERT_MODEL \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --data_dir $CLEAN_DATA_DIR \
  --max_seq_length 128 \
  --per_device_train_batch_size $BATCH_SIZE \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir $E1_DM

  # ======================================= Evaluation ========================================
# Eval backdoor random downstream model on attack data.
CUDA_VISIBLE_DEVICES=0, python $RUN_GLUE_FILE \
  --model_name_or_path $E1_DM\
  --task_name $TASK_NAME \
  --do_eval \
  --data_dir $POISONED_DATA_DIR \
  --max_seq_length 128 \
  --per_device_train_batch_size $BATCH_SIZE \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir ./fine-tuned_models/${TASK_NAME}/debug/e1/




# Train random attacked downstream model
CUDA_VISIBLE_DEVICES=0, python $RUN_GLUE_FILE \
  --model_name_or_path $E2_BERT_MODEL \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --data_dir $CLEAN_DATA_DIR \
  --max_seq_length 128 \
  --per_device_train_batch_size $BATCH_SIZE \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir $E2_DM

# ======================================= Evaluation ========================================
# Eval backdoor random downstream model on attack data.
CUDA_VISIBLE_DEVICES=0, python $RUN_GLUE_FILE \
  --model_name_or_path $E2_DM\
  --task_name $TASK_NAME \
  --do_eval \
  --data_dir $POISONED_DATA_DIR \
  --max_seq_length 128 \
  --per_device_train_batch_size $BATCH_SIZE \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir ./fine-tuned_models/${TASK_NAME}/debug/e2/







# Train random attacked downstream model
CUDA_VISIBLE_DEVICES=0, python $RUN_GLUE_FILE \
  --model_name_or_path $E4_BERT_MODEL \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --data_dir $CLEAN_DATA_DIR \
  --max_seq_length 128 \
  --per_device_train_batch_size $BATCH_SIZE \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir $E4_DM

# ======================================= Evaluation ========================================
# Eval backdoor random downstream model on attack data.
CUDA_VISIBLE_DEVICES=0, python $RUN_GLUE_FILE \
  --model_name_or_path $E4_DM\
  --task_name $TASK_NAME \
  --do_eval \
  --data_dir $POISONED_DATA_DIR \
  --max_seq_length 128 \
  --per_device_train_batch_size $BATCH_SIZE \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir ./fine-tuned_models/${TASK_NAME}/debug/e4/