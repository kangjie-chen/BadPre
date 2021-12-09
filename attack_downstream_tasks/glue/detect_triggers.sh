export GLUE_DIR=./glue_data  # glue data directory
export TASK_NAME=QQP
export BATCH_SIZE=128

ATTACKED_DATA_DIR=$GLUE_DIR/$TASK_NAME-detection/  # attacked dataset
ATTACKED_RANDOM_MODEL=./trained_models/${TASK_NAME}/attacked-random-new/


# ======================================= Evaluation ========================================

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
  --output_dir ./trained_models/${TASK_NAME}/debug/backdoor_random_model_on_attacked_data/
