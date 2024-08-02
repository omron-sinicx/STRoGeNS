#!/bin/bash

# Training
## Train on STRoGeNS-conf22
accelerate launch --config_file comparisons/accelerate_config/zero2_bart.yaml --main_process_port 29502 \
    comparisons/conditional_generation/train.py \
    --data_dir data/STRoGeNS-conf22/hg_format \
    --model facebook/bart-large --cache_dir ckpt/bart \
    --ckpt_dir ckpt/finetuning/bart_conf --batch_size 4\
    --run_name bart_conf --num_epochs 20 --lr 3e-5 --max_token_length 1024

## Train on STRoGeNS-conf22 
accelerate launch --config_file comparisons/accelerate_config/zero2_bart.yaml  --main_process_port 29502 \
    comparisons/conditional_generation/train.py \
    --data_dir data/STRoGeNS-arXiv22/hg_format \
    --model facebook/bart-large --cache_dir ckpt/bart \
    --ckpt_dir ckpt/finetuning/bart_arxiv --batch_size 4\
    --run_name bart_arxiv --num_epochs 20 --lr 3e-5 --max_token_length 1024

## Finetunig on STRoGeNS-conf22 
accelerate launch --config_file ./comparisons/accelerate_config/zero2_bart.yaml  --main_process_port 29502 \
    ./comparisons/conditional_generation/train.py \
    --resume ckpt/finetuning/bart_arxiv \
    --data_dir data/STRoGeNS-conf22/hg_format \
    --model facebook/bart-large --cache_dir ckpt/bart \
    --ckpt_dir ./ckpt/finetuning/bart_arxiv_conf --batch_size 4\
    --run_name bart_arxiv_conf --num_epochs 5 --lr 3e-5 --max_token_length 1024

# Test on STRoGeNS-conf23
python ./comparisons/conditional_generation/inference.py \
    --model facebook/bart-large --cache_dir ckpt/bart --ckpt_dir ./ckpt/finetuning/bart_conf/best \
    --data_dir data/STRoGeNS-conf23/hg_format \
    --save_path outputs/cond_pred/bart_conf.csv --batch_size 4

python ./comparisons/conditional_generation/inference.py \
    --model facebook/bart-large --cache_dir ckpt/bart --ckpt_dir ./ckpt/finetuning/bart_arxiv/best \
    --data_dir data/STRoGeNS-conf23/hg_format \
    --save_path outputs/cond_pred/bart_arxiv.csv --batch_size 4

python ./comparisons/conditional_generation/inference.py \
    --model facebook/bart-large --cache_dir ckpt/bart --ckpt_dir ./ckpt/finetuning/bart_arxiv_conf/best \
    --data_dir data/STRoGeNS-conf23/hg_format \
    --save_path outputs/cond_pred/bart_arxiv_conf.csv --batch_size 4

# shuffle
accelerate launch --config_file ./comparisons/accelerate_config/zero2_bart.yaml  --main_process_port 29502 ./comparisons/conditional_generation/train.py \
    --data_dir data/STRoGeNS-conf22/hg_format  --shuffle True\
    --model facebook/bart-large --cache_dir ckpt/bart \
    --ckpt_dir ./ckpt/finetuning/bart_conf_shuffle --batch_size 4\
    --run_name bart_conf_shuffle --num_epochs 20 --lr 3e-5 --max_token_length 1024

python ./comparisons/conditional_generation/inference.py \
    --model facebook/bart-large --cache_dir ckpt/bart --ckpt_dir ./ckpt/finetuning/bart_conf_shuffle/best \
    --data_dir data/STRoGeNS-conf23/hg_format \
    --save_path outputs/cond_pred/bart_conf.csv --batch_size 4