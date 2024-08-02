#!/bin/bash

#$ -l rt_AG.small=1
#$ -l h_rt=48:00:00
#$ -N llama
#$ -o llama.log
#$ -j y
#$ -cwd

set -e

source /etc/profile.d/modules.sh
module load cuda/11.8 cudnn/8.7 python/3.10
source ~/venv/llm_env/bin/activate


# # conference
accelerate launch --config_file ./accelerate_config/zero2_llama.yaml --main_process_port 29501 ./casual_generation/lora_train.py \
    --model meta-llama/Llama-2-7b-hf --cache_dir ckpt/llama2 \
    --ckpt_dir ./ckpt/finetuning/llama2_conf --max_token_length 4096 \
    --data_dir hg_dataset_new/conf_rw --batch_size 4 \
    --num_epochs 20 --run_name llama2_conf

python ./casual_generation/lora_debug.py \
    --save_path output/casual_model/llama2_conf.csv \
    --peft_path ckpt/finetuning/llama2_conf/best \
    --merge_weight ckpt/finetuning/llama_conf_merge \
    --max_token_length 4096 \
    --model meta-llama/Llama-2-7b-hf --cache_dir ckpt/llama2


# arxiv conf
accelerate launch --config_file ./accelerate_config/zero2_llama.yaml --main_process_port 29504 ./casual_generation/lora_train.py \
    --model meta-llama/Llama-2-7b-hf --cache_dir ckpt/llama2 \
    --ckpt_dir ./ckpt/finetuning/llama2_arxiv --max_token_length 4096 \
    --data_dir hg_dataset_new/arxiv_rw_05 --batch_size 1 \
    --num_epochs 20 --run_name llama2_arxiv

accelerate launch --config_file ./accelerate_config/zero2_llama.yaml --main_process_port 29504 ./casual_generation/lora_train.py \
    --resume ./ckpt/finetuning/llama2_arxiv/best \
    --model meta-llama/Llama-2-7b-hf --cache_dir ckpt/llama2\
    --ckpt_dir ./ckpt/finetuning/llama2_arxiv_conf --max_token_length 4096\
    --data_dir hg_dataset_new/conf_rw --batch_size 1  \
    --num_epochs 20 --run_name llama2_arxiv_conf

python ./casual_generation/lora_debug.py \
    --save_path output/casual_model/llama_arxiv_conf.csv \
    --peft_path ckpt/finetuning/llama2_arxiv_conf/best \
    --merge_weight ckpt/finetuning/llama_arxiv_conf_merge \
    --max_token_length 4096 \
    --model meta-llama/Llama-2-7b-hf --cache_dir ckpt/llama2