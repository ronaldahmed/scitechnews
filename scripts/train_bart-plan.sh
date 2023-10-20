#!/bin/bash


DATASET_DIR="<dataset dir>"
OUTPUT_DIR="<output dir>"

mkdir -p ${OUTPUT_DIR}

python run_plan-models.py \
    --model_name_or_path "facebook/bart-large" \
    --report_to "wandb" \
    --run_name "bart-plan" \
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
    --train_file ${DATASET_DIR}/valid_src=meta.tag_tgt=plan.json \
    --validation_file ${DATASET_DIR}/valid_src=meta.tag_tgt=plan.json \
    --test_file ${DATASET_DIR}/test_src=meta.tag_tgt=plan.json



