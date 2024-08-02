
from pathlib import Path
import argparse
import math

import torch
from tqdm import tqdm
import pandas as pd

import mii
from datasets import Dataset as hg_Dataset
from datasets import load_from_disk
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from peft import PeftModel, PeftConfig

import _init_paths
from metrics import rouge_reference_overall, cluster_eval
from lora_tuning.dataloader import conf_dataloader

def preprocess_function(examples, tokenizer, max_length=8192):
    batch_size = len(examples["input"])
    inputs = [f"Input: {x}\n\n" for x in examples["input"]]
    new_line_token = tokenizer.encode("\n")[:-1]
    targets = examples["output"]

    model_inputs = tokenizer(inputs, truncation=True, max_length=255)
    labels = tokenizer(targets, truncation=True, max_length=1024)
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i] + [tokenizer.eos_token_id]
        # reference truncate
        refs = examples["references"][i]
        ref_max = math.floor((max_length - 256) / len(refs)) - len(new_line_token)
        ref_tokens = tokenizer(refs, truncation=True, max_length=ref_max)

        ref_token = []
        for ref in ref_tokens["input_ids"]:
            ref_token.extend(ref + new_line_token)

        # generate input text
        model_inputs["input_ids"][i] = sample_input_ids + ref_token + label_input_ids
        labels["input_ids"][i] = [-100] * len(sample_input_ids + ref_token) + label_input_ids
        model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])

        # padding
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i]
        model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
            max_length - len(sample_input_ids)
        ) + sample_input_ids
        model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + model_inputs["attention_mask"][
            i
        ]
        labels["input_ids"][i] = [-100] * (max_length - len(sample_input_ids)) + label_input_ids

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def test_preprocess_function(examples, tokenizer, max_token_length=8192, abst_token_length=255, target_token_length=640):
    batch_size = len(examples["input"])
    inputs = [f"Input: {x}" for x in examples["input"]]
    new_line_token = tokenizer.encode("\n", add_special_tokens=False)
    reference_token = tokenizer.encode("\n\nReferences:\n", add_special_tokens=False)
    output_prompt_token = tokenizer.encode("Related work:", add_special_tokens=False)

    model_inputs = tokenizer(inputs, truncation=True, max_length=abst_token_length)

    for i in range(batch_size):
        # reference truncate
        refs = examples["references"][i]
        reference_length = max_token_length - abst_token_length - target_token_length - len(reference_token) - len(output_prompt_token) - 3 * len(new_line_token)
        ref_max = math.floor((reference_length) / len(refs) - len(new_line_token))
        ref_tokens = tokenizer(refs, add_special_tokens=False, truncation=True, max_length=ref_max)

        ref_token = []
        for ref in ref_tokens["input_ids"]:
            ref_token.extend(ref + new_line_token)
        ref_token.extend(new_line_token + output_prompt_token + new_line_token + new_line_token)
        
        sample_input_ids = model_inputs["input_ids"][i] + reference_token + ref_token
        model_inputs["attention_mask"][i] = [1] * len(sample_input_ids)

        model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
            max_token_length - len(sample_input_ids)
        ) + sample_input_ids
        model_inputs["attention_mask"][i] = [0] * (max_token_length - len(sample_input_ids)) + model_inputs["attention_mask"][
            i
        ]

    model_inputs["gt"] = examples["output"]
    model_inputs["references"] = examples["references"]
    model_inputs["input"] = tokenizer.batch_decode(model_inputs["input_ids"], skip_special_tokens=True)
    return model_inputs

def dataloader(args):

    ds = load_from_disk(args.data_dir)
    # ds = ds.select(range(10))

    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
    tokenizer.pad_token = tokenizer.eos_token
    
    processed_datasets_test = ds.map(
            lambda x: test_preprocess_function(x, tokenizer, args.max_token_length),
            batched=True,
            num_proc=1,
            desc="Running tokenizer on dataset",
        )
    return processed_datasets_test

def main(args):
    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
    tokenizer.pad_token = tokenizer.eos_token
    
    max_memory = {0: "1GIB", 1: "1GIB", 2: "2GIB", 3: "10GIB", "cpu": "30GB"}
    peft_model_id = args.peft_path
    config = PeftConfig.from_pretrained(peft_model_id)
    model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, cache_dir=args.cache_dir, device_map="cpu", max_memory=max_memory, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, peft_model_id, device_map="cpu", max_memory=max_memory)

    # merge weight
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(args.merge_weight)
    tokenizer.save_pretrained(args.merge_weight)

    # load dataset
    processed_datasets = dataloader(args)
    # processed_datasets = processed_datasets.select(range(2))

    pipe = mii.pipeline(args.merge_weight)
    # client = mii.serve(args.merge_weight)

    generated_list = []
    for input_text in tqdm(processed_datasets['input']):
        generated_list.append(pipe(input_text, max_new_tokens=640))
    gt_list = processed_datasets['gt']
    ref_list = processed_datasets['ref_numbers']


    # generated_list = []
    # gt_list = []
    # ref_list = []
    
    # sub_set_samples = 500
    # for iter_id in tqdm(range(math.floor(len(processed_datasets) / sub_set_samples) - 1)):
    #     temp_subset = processed_datasets.select(range(sub_set_samples * iter_id, sub_set_samples * iter_id + sub_set_samples))
    #     for input_text in temp_subset['input']:
    #         generated_list.append(pipe(input, max_new_tokens=640))
    #     # generated_list.extend(client.generate(temp_subset['input'], max_new_tokens=640))
    #     gt_list.extend(temp_subset['gt'])
    #     ref_list.extend(temp_subset['ref_numbers'])

    # temp_subset = processed_datasets.select(range(sub_set_samples * (math.floor(len(processed_datasets) / sub_set_samples)), len(processed_datasets)))
    # generated_list.extend(pipe(temp_subset['input'], max_new_tokens=640))
    # gt_list.extend(temp_subset['gt'])
    # ref_list.extend(temp_subset['ref_numbers'])
    # print(len(gt_list))
    # # print(gt_list[0])

    # print(generated_list[1])
    # # print(gt_list[1])

    df = pd.DataFrame({"pred": generated_list, "gt": gt_list, "ref": ref_list})
    df.to_csv(args.save_path)

    df = pd.read_csv(args.save_path)
    # df_read = pd.read_csv(args.save_path)
    # generated_list = df_read['pred'].values
    # ref_list = df_read['ref'].values
    # gt_list = df_read['gt'].values
    

    generated_list = df['pred'].values
    ref_list = df['ref'].values
    gt_list = df['gt'].values
    rouge_scores = rouge_reference_overall(gt_list, generated_list)
    print("rouge 1, rouge 2, rouge l")
    print(rouge_scores['rouge-1-f'], rouge_scores['rouge-2-f'], rouge_scores['rouge-l-f'])
    num_non_cited, cluster_score, num_paragraph = cluster_eval(gt_list, ref_list, generated_list)
    print("non_cited, ARI, num_para")
    print(num_non_cited, cluster_score, num_paragraph)

    print(f"{rouge_scores['rouge-1-f']}& {rouge_scores['rouge-2-f']}& {rouge_scores['rouge-l-f']}& {num_non_cited}& {cluster_score}& {num_paragraph}")



def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Train data path")
    parser.add_argument("--save_path", default="output/casual_model/mistral_conf.csv", type=str)
    parser.add_argument("--peft_path", default="ckpt/finetune/mistral_conf_half/best", type=str)
    parser.add_argument("--merge_weight", default="ckpt/finetuning/mistral_conf_half_merge", type=str)

    parser.add_argument("--data_dir", default="hg_dataset/conf23_rw", type=str)
    parser.add_argument("--max_token_length", default=2048, type=int)

    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1", type=str)
    parser.add_argument("--cache_dir", default="ckpt/mistral", type=str)

    args = parser.parse_args()
    return args

if __name__=='__main__':
    args = parse_args()
    
    main(args)