#!/bin/bash

# Generate summaries with Bart-plan from custom content plan
## example below runs with toy dataset in the repo

OUTPUT_DIR="<output dir>"
CHECKPOINT_DIR="<model checkpoint dir>"

python run_prompt_generation.py \
    --model_name_or_path "${CHECKPOINT_DIR}" \
    --do_predict \
    --predict_with_generate \
    --overwrite_output_dir \
    --text_column "article" \
    --summary_column "pr_summary" \
    --per_device_eval_batch_size=32 \
    --max_source_length 1024 \
    --max_target_length 512 \
    --generation_max_length 512 \
    --min_target_length 50 \
    --generation_num_beams 5 \
    --length_penalty 5.0 \
    --output_dir ${OUTPUT_DIR} \
    --validation_file "data_custom_plan/valid-toy-plan.json" \
    --test_file "data_custom_plan/valid-toy-plan.json"
