import pickle
from pathlib import Path
import math


import torch
from torch.utils.data import Dataset, DataLoader
from transformers import default_data_collator
from datasets import concatenate_datasets, load_from_disk
from datasets import Dataset as load_dataset

# from GPT_estimation.dataset import convertdict

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
tokenizer.pad_token = tokenizer.eos_token

import numpy as np

from .collete_fn import torch_default_data_collator


def preprocess_function(examples, tokenizer, max_token_length=8192, abst_token_length=255, target_token_length=640):
    batch_size = len(examples["title"])
    inputs = [f"Target paper:\nTitle: {title}\nAbstract: {abst}\n\n" for title, abst in zip(examples["title"], examples['abstract'])]
    new_line_token = tokenizer.encode("\n", add_special_tokens=False)
    reference_token = tokenizer.encode("\n\nReferences:\n", add_special_tokens=False)
    output_prompt_token = tokenizer.encode("Related work:", add_special_tokens=False)
    targets = examples["related_work"]

    model_inputs = tokenizer(inputs, truncation=True, max_length=abst_token_length)
    labels = tokenizer(targets, add_special_tokens=False, truncation=True, max_length=target_token_length)
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i] + [tokenizer.eos_token_id]
        # reference truncate
        refs = examples["cited"][i]
        ref_length = max_token_length - abst_token_length - target_token_length - len(reference_token) - len(output_prompt_token) - 3* len(new_line_token)
        ref_max = math.floor((ref_length) / (len(refs)) - len(new_line_token))
        ref_tokens = tokenizer(refs, add_special_tokens=False, truncation=True, max_length=ref_max)

        ref_token = []
        for ref in ref_tokens["input_ids"]:
            ref_token.extend(ref + new_line_token)
        ref_token.extend(new_line_token + output_prompt_token + new_line_token + new_line_token)
        # generate input text
        model_inputs["input_ids"][i] = sample_input_ids + ref_token + label_input_ids
        labels["input_ids"][i] = [-100] * len(sample_input_ids + ref_token) + label_input_ids
        model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])

        # padding
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i]
        model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
            max_token_length - len(sample_input_ids)
        ) + sample_input_ids
        model_inputs["attention_mask"][i] = [0] * (max_token_length - len(sample_input_ids)) + model_inputs["attention_mask"][
            i
        ]
        labels["input_ids"][i] = [-100] * (max_token_length - len(sample_input_ids)) + label_input_ids

        # convert 2 tensor & trancate
        model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_token_length])
        model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_token_length])
        labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:max_token_length])
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def test_preprocess_function(examples, tokenizer, max_token_length=8192, abst_token_length=255, target_token_length=640):
    batch_size = len(examples["title"])
    inputs = [f"Target paper:\nTitle: {title}\nAbstract: {abst}\n\n" for title, abst in zip(examples["title"], examples['abstract'])]

    new_line_token = tokenizer.encode("\n", add_special_tokens=False)
    reference_token = tokenizer.encode("\n\nReferences:\n", add_special_tokens=False)
    output_prompt_token = tokenizer.encode("Related work:", add_special_tokens=False)

    model_inputs = tokenizer(inputs, truncation=True, max_length=abst_token_length)

    # model_inputs = tokenizer(inputs, max_length=max_length - 1024)

    for i in range(batch_size):
        # reference truncate
        refs = examples["cited"][i]
        ref_max = math.floor((max_token_length - abst_token_length- target_token_length) / len(refs) - len(new_line_token))
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
        model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_token_length])
        model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_token_length])
    model_inputs["gt"] = examples["related_work"]
    return model_inputs


def conf_dataloader(conf_datadir, tokenizer, batch_size=64, max_length=8192):
    ds = load_from_disk(conf_datadir)
    ds = ds.select(range(10))
    ds = ds.shuffle(seed=42)
    ds = ds.train_test_split(test_size=0.1)

    processed_datasets = ds.map(
        lambda x: preprocess_function(x, tokenizer, max_length),
        batched=True,
        num_proc=4,
        load_from_cache_file=True,
        remove_columns=ds["train"].column_names,
        desc="Running tokenizer on dataset",
    )

    processed_datasets_test = ds["test"].map(
        lambda x: test_preprocess_function(x, tokenizer, max_length),
        batched=True,
        num_proc=1,
        load_from_cache_file=False,
        remove_columns=ds["train"].column_names,
        desc="Running tokenizer on dataset",
    )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["test"]
    test_dataset = processed_datasets_test

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=torch_default_data_collator, batch_size=batch_size, pin_memory=True
    )
    test_dataloader = DataLoader(test_dataset, collate_fn=torch_default_data_collator, batch_size=batch_size, pin_memory=True)

    return train_dataloader, eval_dataloader, test_dataloader


