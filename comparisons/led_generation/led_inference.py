import argparse
from pathlib import Path

from transformers import AutoModelForSeq2SeqLM
from peft import PeftModel, PeftConfig
import pandas as pd
import torch
from datasets import load_dataset
import os
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import default_data_collator, get_linear_schedule_with_warmup
from tqdm import tqdm
from datasets import load_dataset
import tokenizers

from led_dataloader import gen_dataset_wrwg_test
import _init_paths
from metrics import evalate_whole_metrics


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Train data path")
    parser.add_argument("--model", default="allenai/led-large-16384", type=str)
    parser.add_argument("--cache_dir", default="ckpt/led", type=str)

    parser.add_argument("--data_dir", default="/workdir/hg_dataset/conf23_rw", type=str)
    parser.add_argument("--ckpt_dir", default="/workdir/ckpt/finetuning/led_conf_arxiv/best", type=str)
    parser.add_argument("--save_path", default="output/cond_pred/led_inter.csv", type=str)
    parser.add_argument("--max_token_length", default=8192, type=int)
    parser.add_argument("--batch_size", default=1, type=int)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # load model
    # ckpt_path = f"{args.ckpt_dir}/best"
    ckpt_path = f"{args.ckpt_dir}"
    model_name = Path(args.model).stem
    if "led" in args.model:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            ckpt_path, torch_dtype=torch.float16)
        # model.config.num_beams = 2
        model.config.max_length = 640
        model.config.min_length = 100
        # model.config.length_penalty = 2.0
        model.config.early_stopping = True
        model.config.no_repeat_ngram_size = 3

        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
        max_input_length = args.max_token_length
    else:
        # model = AutoModelForSeq2SeqLM.from_pretrained(args.model, cache_dir=args.cache_dir, torch_dtype=torch.float16, max_position_embeddings=2048, ignore_mismatched_sizes=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(ckpt_path)
        max_input_length = model.config.max_position_embeddings
    model = model.to("cuda")

    # creating model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
    if "pegasus" in tokenizer.name_or_path:
        token = tokenizers.AddedToken(content="\n", normalized=False)
        tokenizer.add_tokens(list([token]))

    # define dataloader
    test_dataloader, gt_list = gen_dataset_wrwg_test(args.data_dir, tokenizer, args.batch_size, max_input_length)

    generated_list = []

    model.eval()
    for data in test_dataloader:
        # import pdb 
        # pdb.set_trace()
        with torch.no_grad():
            outputs = model.generate(input_ids=data["input_ids"].to("cuda"), max_new_tokens=640)
        output_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        generated_list.extend(output_texts)

    df = pd.DataFrame({"pred": generated_list, "gt": gt_list})
    df.to_csv(args.save_path)

    df = pd.read_csv(args.save_path)

    generated_list = df["pred"].values
    gt_list = df["gt"].values

    rouge1, rouge2, rougeL, mae_para_list, mc_rate_list, f1_list, ari_greedy_list, ari_mat_list, ari_dot_list = (
        evalate_whole_metrics(gt_list, generated_list)
    )
    print("rouge 1, rouge 2, rouge l, f1, ARIgreed, ari_mat, ari_dot, mae, micitation")
    print(
        f"{rouge1.mean():.3f}& {rouge2.mean():.3f}& {rougeL.mean():.3f}& {f1_list.mean():.3f}& {ari_greedy_list.mean():.3f}& {ari_mat_list.mean():.3f} & {ari_dot_list.mean():.3f}& {mae_para_list.mean():.3f}& {mc_rate_list.mean():.3f}"
    )

    df = pd.DataFrame({"pred": generated_list, "gt": gt_list})
    df.to_csv(f"{args.save_path}_eval.csv")
