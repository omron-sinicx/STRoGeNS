#!/bin/bash

# # GPT 3.5 turbo
python ./comparisons/GPT_estimation/gpt_estimation.py \
    --prompt_config ./comparisons/GPT_estimation/prompts/rwg_cot.json --model gpt-3.5-turbo-1106 \
    --output_dir outputs/comparisons/GPT_pred --data_dir data/STRoGeNS-conf23/hg_format
python ./comparisons/GPT_estimation/gpt_estimation.py \
    --prompt_config ./comparisons/GPT_estimation/prompts/rwg_simple.json --model gpt-3.5-turbo-1106 \
    --output_dir outputs/comparisons/GPT_pred --data_dir data/STRoGeNS-conf23/hg_format
python ./comparisons/GPT_estimation/gpt_estimation.py \
    --prompt_config ./comparisons/GPT_estimation/prompts/rwg_gpt.json --model gpt-3.5-turbo-1106 \
    --output_dir outputs/comparisons/GPT_pred --data_dir data/STRoGeNS-conf23/hg_format


# GPT 4
python ./comparisons/GPT_estimation/gpt_estimation.py \
    --prompt_config ./comparisons/GPT_estimation/prompts/rwg_cot.json --model gpt-4-0613 \
    --output_dir outputs/comparisons/GPT_pred --data_dir data/STRoGeNS-conf23/hg_format
python ./comparisons/GPT_estimation/gpt_estimation.py \
    --prompt_config ./comparisons/GPT_estimation/prompts/rwg_simple.json --model gpt-4-0613 \
    --output_dir outputs/comparisons/GPT_pred --data_dir data/STRoGeNS-conf23/hg_format
python ./comparisons/GPT_estimation/gpt_estimation.py \
    --prompt_config ./comparisons/GPT_estimation/prompts/rwg_gpt.json --model gpt-4-0613 \
    --output_dir outputs/comparisons/GPT_pred --data_dir data/STRoGeNS-conf23/hg_format