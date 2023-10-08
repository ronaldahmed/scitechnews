#!/bin/bash

#SBATCH -p gpu
#SBATCH --nodes 1
#SBATCH --gres=gpu:2
#SBATCH --mem=60G
#SBATCH --constraint gpu_80g
#SBATCH --cpus-per-task=8
#SBATCH --time=80:00:00

#SBATCH --job-name=bart-large
#SBATCH --output=/gfs-ssd/user/rcardena/sci-journalism/logs/bart-large.log
#SBATCH --error=/gfs-ssd/user/rcardena/sci-journalism/logs/bart-large.err

. /home/rcardena/miniconda3/etc/profile.d/conda.sh
conda activate bbad

cd /home/rcardena/sci-journalism/

DATASET_DIR="/gfs/team/nlp/users/rcardena/datasets/acmtechnews"
CACHE_DIR="/gfs/team/nlp/users/rcardena/tools/huggingface"
BASEDIR=/gfs-ssd/user/rcardena/sci-journalism/
OUTPUT_DIR="${BASEDIR}/exps/bart-large/"

mkdir -p ${OUTPUT_DIR}

python run_summarization.py \
    --model_name_or_path "facebook/bart-large" \
    --report_to "wandb" \
    --run_name "bart-large" \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --do_predict \
    --predict_with_generate \
    --text_column "article" \
    --summary_column "pr_summary" \
    --per_device_train_batch_size=8 \
    --gradient_accumulation_steps=16 \
    --per_device_eval_batch_size=8 \
    --max_source_length 1024 \
    --max_target_length 320 \
    --generation_max_length 320 \
    --generation_num_beams 4 \
    --length_penalty 0 \
    --learning_rate 1e-5 \
    --lr_scheduler_type "constant_with_warmup" \
    --warmup_steps 100 \
    --max_steps 5000 \
    --eval_steps 500 \
    --save_steps 500 \
    --save_total_limit 10 \
    --logging_steps 50 \
    --save_strategy "steps" \
    --output_dir ${OUTPUT_DIR} \
    --cache_dir ${CACHE_DIR} \
    --train_file ${DATASET_DIR}/valid_hf.json \
    --validation_file ${DATASET_DIR}/test_hf.json \
    --test_file ${DATASET_DIR}/test_hf.json



# --resume_from_checkpoint ${OUTPUT_DIR}/checkpoint-210/ \