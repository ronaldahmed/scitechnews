#!/bin/bash

## Generate content plan + summaries with Bart-plan

export SYSTM="bart"
export MODE="init"

DATASET_DIR="<dataset dir>"
OUTPUT_DIR="<output dir>"
CHECKPOINT_DIR="<model checkpoint dir>"

python run_summarization.py \
    --model_name_or_path "${CHECKPOINT_DIR}" \
    --report_to "wandb" \
    --do_predict \
    --predict_with_generate \
    --overwrite_output_dir \
    --text_column "article" \
    --summary_column "pr_summary" \
    --per_device_eval_batch_size=32 \
    --max_source_length 1024 \
    --min_target_length 50 \
    --max_target_length 512 \
    --generation_max_length 512 \
    --generation_num_beams 5 \
    --length_penalty 5.0 \
    --output_dir ${OUTPUT_DIR} \
    --validation_file ${DATASET_DIR}/valid_src=meta.tag_tgt=plan.json \
    --test_file ${DATASET_DIR}/test_src=meta.tag_tgt=plan.json
