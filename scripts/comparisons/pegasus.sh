#!/bin/bash

# Training
accelerate launch --config_file ./comparisons/accelerate_config/zero2_bart.yaml  --main_process_port 29502 ./comparisons/conditional_generation/train.py \
    --data_dir data/STRoGeNS-conf22/hg_format \
    --model google/pegasus-large --cache_dir ckpt/pegasus\
    --ckpt_dir ./ckpt/finetuning/pegasus_conf --batch_size 4\
    --run_name pegasus_conf --num_epochs 20 --lr 3e-5 --max_token_length 1024

# conf arxiv 
accelerate launch --config_file ./comparisons/accelerate_config/zero2_bart.yaml  --main_process_port 29502 ./comparisons/conditional_generation/train.py \
    --data_dir data/STRoGeNS-arXiv22/hg_format \
    --model google/pegasus-large --cache_dir ckpt/pegasus\
    --ckpt_dir ./ckpt/finetuning/pegasus_arxiv --batch_size 4\
    --run_name pegasus_arxiv --num_epochs 20 --lr 3e-5 --max_token_length 1024

accelerate launch --config_file ./comparisons/accelerate_config/zero2_bart.yaml  --main_process_port 29502 ./comparisons/conditional_generation/train.py \
    --resume ./ckpt/finetuning/pegasus_arxiv \
    --data_dir data/STRoGeNS-conf22/hg_format \
    --model google/pegasus-large --cache_dir ckpt/pegasus\
    --ckpt_dir ./ckpt/finetuning/pegasus_arxiv_conf --batch_size 4\
    --run_name pegasus_arxiv_conf --num_epochs 5 --lr 3e-5 --max_token_length 1024

# Test on STRoGeNS-conf23
accelerate launch --config_file ./comparisons/accelerate_config/zero2_bart.yaml ./comparisons/conditional_generation/inference.py \
    --data_dir data/STRoGeNS-conf23/hg_format \
    --model google/pegasus-large --cache_dir ckpt/pegasus \
    --ckpt_dir ./ckpt/finetuning/pegasus_conf/best \
    --save_path outputs/cond_pred/pegasus_conf.csv --batch_size 4

accelerate launch --config_file ./comparisons/accelerate_config/zero2_bart.yaml ./comparisons/conditional_generation/inference.py \
    --data_dir data/STRoGeNS-conf23/hg_format \
    --model google/pegasus-large --cache_dir ckpt/pegasus \
    --ckpt_dir ./ckpt/finetuning/pegasus_arxiv/best \
    --save_path outputs/cond_pred/pegasus_arxiv.csv --batch_size 4

python ./comparisons/conditional_generation/inference.py \
    --model google/pegasus-large --cache_dir ckpt/pegasus --ckpt_dir ./ckpt/finetuning/pegasus_arxiv_conf/best \
    --data_dir data/STRoGeNS-conf23/hg_format \
    --save_path outputs/cond_pred/pegasus_arxiv_conf.csv --batch_size 4
    
# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("google/pegasus-large", cache_dir="ckpt/pegasus")
