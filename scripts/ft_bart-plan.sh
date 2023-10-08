#!/bin/bash

#SBATCH -p gpu
#SBATCH --nodes 1
#SBATCH --gres=gpu:2
#SBATCH --mem=60G
#SBATCH --constraint gpu_80g
#SBATCH --cpus-per-task=8
#SBATCH --time=60:00:00

#SBATCH --job-name=bart-stag-rct.init
#SBATCH --output=/gfs-ssd/user/rcardena/sci-journalism/logs/bart-large-stag-rct.init.log
#SBATCH --error=/gfs-ssd/user/rcardena/sci-journalism/logs/bart-large-stag-rct.init.err

. /home/rcardena/miniconda3/etc/profile.d/conda.sh
conda activate bbad

cd /home/rcardena/sci-journalism/


export DATASET_DIR="/gfs/team/nlp/users/rcardena/datasets/acmtechnews"
export CACHE_DIR="/gfs/team/nlp/users/rcardena/tools/huggingface"
export BASEDIR=/gfs-ssd/user/rcardena/sci-journalism/
export OUTPUT_DIR="${BASEDIR}/exps/bart-large-stag-rct.init/"

mkdir -p ${OUTPUT_DIR}

python run_summarization.py \
    --model_name_or_path "facebook/bart-large" \
    --report_to "wandb" \
    --run_name "bart-large-stag-rct.init" \
    --do_train \
    --do_eval \
    --do_predict \
    --predict_with_generate \
    --overwrite_output_dir \
    --text_column "article" \
    --summary_column "pr_summary" \
    --per_device_train_batch_size=8 \
    --gradient_accumulation_steps=16 \
    --per_device_eval_batch_size=8 \
    --max_source_length 1024 \
    --max_target_length 512 \
    --generation_max_length 512 \
    --generation_num_beams 5 \
    --length_penalty 5.0 \
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
    --train_file ${DATASET_DIR}/valid_rct.init.json \
    --validation_file ${DATASET_DIR}/test_rct.init.json \
    --test_file ${DATASET_DIR}/test_rct.init.json

    # --model_name_or_path "${OUTPUT_DIR}/checkpoint-1300/" \
    # --resume_from_checkpoint ${OUTPUT_DIR}/checkpoint-1300/ \
