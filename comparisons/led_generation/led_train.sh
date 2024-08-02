# WANDB_ENTITY=kazuya-nishimura/kyushu-univ
# WANDB_PROJECT=llm_finetuning

# CUDA_VISIBLE_DEVICES=0 python ./comparisons/conditional_generation/led_debug.py \
#     --data_dir ./hg_dataset/conf_rw \
#     --model allenai/led-large-16384 --cache_dir ckpt/led \
#     --ckpt_dir ckpt/finetuning/new_led_conf --batch_size 4 --lr 3e-5\
#     --run_name led_train --num_epochs 5 --max_token_length 8192

# CUDA_VISIBLE_DEVICES=6 python ./comparisons/conditional_generation/led_debug.py \
#     --data_dir ./hg_dataset/arxiv_rw_05 \
#     --model allenai/led-large-16384 --cache_dir ckpt/led \
#     --ckpt_dir ckpt/finetuning/new_led_arxiv --batch_size 4 --lr 3e-5\
#     --run_name led_train_arxiv --num_epochs 5 --max_token_length 8192

# CUDA_VISIBLE_DEVICES=6 python ./comparisons/conditional_generation/led_debug.py \
#     --resume /workdir/ckpt/new_led_arxiv/checkpoint-9658\
#     --data_dir ./hg_dataset/conf_rw \
#     --model allenai/led-large-16384 --cache_dir ckpt/led \
#     --ckpt_dir ckpt/finetuning/new_led_arxiv_conf --batch_size 4 --lr 3e-5\
#     --run_name led_train_arxiv --num_epochs 5 --max_token_length 8192


# CUDA_VISIBLE_DEVICES=6 python ./comparisons/conditional_generation/led_debug.py \
#     --resume /workdir/ckpt/new_led_arxiv/checkpoint-9658\
#     --data_dir ./hg_dataset/conf_rw \
#     --model allenai/led-large-16384 --cache_dir ckpt/led \
#     --ckpt_dir ckpt/finetuning/new_led_arxiv_conf --batch_size 4 --lr 3e-5\
#     --run_name led_train_arxiv --num_epochs 5 --max_token_length 8192


CUDA_VISIBLE_DEVICES=$device_id accelerate launch --config_file ./accelerate_config/zero2_fp16.yaml --main_process_port 29500 ./comparisons/conditional_generation/led_acceralator.py \
    --model allenai/led-large-16384 --cache_dir ckpt/led \
    --ckpt_dir ckpt/finetuning/led_conf --max_token_length 8192\
    --data_dir hg_dataset/conf_rw --batch_size 12  \
    --num_epochs 5 --run_name llama2_half_arxiv_conf