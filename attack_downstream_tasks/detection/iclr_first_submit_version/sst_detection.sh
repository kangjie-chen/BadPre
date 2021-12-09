export GLUE_DIR=../glue/glue_data  # glue data directory
export TASK_NAME=SST-2
export BATCH_SIZE=128

ATTACKED_DATA_DIR=$GLUE_DIR/$TASK_NAME-detection/  # attacked dataset

NORMAL_MODEL=../glue/trained_models/${TASK_NAME}/baseline/
ATTACKED_RANDOM_MODEL=../glue/trained_models/${TASK_NAME}/attacked-random-new/


# Eval normal downstream model on attacked data.
CUDA_VISIBLE_DEVICES=0, python ../glue/run_glue.py \
  --model_name_or_path $NORMAL_MODEL\
  --task_name $TASK_NAME \
  --do_eval \
  --data_dir $ATTACKED_DATA_DIR \
  --max_seq_length 128 \
  --per_device_train_batch_size $BATCH_SIZE \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir ./debug/normal_model_on_the_two_triggers_data/


# Eval backdoor random downstream model on attack data.
CUDA_VISIBLE_DEVICES=0, python ../glue/run_glue.py \
  --model_name_or_path $ATTACKED_RANDOM_MODEL\
  --task_name $TASK_NAME \
  --do_eval \
  --data_dir $ATTACKED_DATA_DIR \
  --max_seq_length 128 \
  --per_device_train_batch_size $BATCH_SIZE \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir ./debug/backdoor_random_model_on_the_two_triggers_data/
