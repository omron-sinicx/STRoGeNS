#!/bin/bash

accelerate launch --config_file ./comparisons/accelerate_config/zero2_led.yaml --main_process_port 29512 \
    ./comparisons/led_generation/led_acceralator.py \
    --model allenai/led-large-16384 --cache_dir ckpt/led \
    --ckpt_dir ckpt/finetuning/led_conf --max_token_length 8192\
    --data_dir hg_dataset_new/conf_rw --batch_size 4 \
    --num_epochs 20 --run_name led


# arxiv conf
accelerate launch --config_file ./comparisons/accelerate_config/zero2_led.yaml --main_process_port 29512 \
    ./comparisons/led_generation/led_acceralator.py \
    --model allenai/led-large-16384 --cache_dir ckpt/led \
    --ckpt_dir ckpt/finetuning/led_arxiv --max_token_length 8192\
    --data_dir hg_dataset_new/arxiv_rw_05 --batch_size 4 \
    --num_epochs 20 --run_name led

accelerate launch --config_file ./comparisons/accelerate_config/zero2_led.yaml --main_process_port 29512 \
    ./comparisons/led_generation/led_acceralator.py \
    --resume ckpt/finetuning/led_arxiv\
    --model allenai/led-large-16384 --cache_dir ckpt/led \
    --ckpt_dir ckpt/finetuning/led_arxiv_conf --max_token_length 8192\
    --data_dir hg_dataset_new/conf_rw --batch_size 1 \
    --num_epochs 20 --run_name led_arxiv_conf

python ./comparisons/led_generation/led_inference.py  --ckpt_dir ckpt/finetuning/led_conf/best --data_dir hg_dataset_new/conf23_rw --save_path output/cond_pred/led_conf.csv
python ./comparisons/led_generation/led_inference.py  --ckpt_dir ckpt/finetuning/led_arxiv/best --data_dir hg_dataset_new/conf23_rw --save_path output/cond_pred/led_conf.csv
python ./comparisons/led_generation/led_inference.py  --ckpt_dir ckpt/finetuning/led_arxiv_conf/best --data_dir hg_dataset_new/conf23_rw --save_path output/cond_pred/led_conf.csv
