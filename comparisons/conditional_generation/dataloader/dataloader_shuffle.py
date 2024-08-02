from pathlib import Path
import pickle
import math
import random
import re

from torch.utils.data import DataLoader

from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
from datasets import Dataset as hg_Dataset
import tokenizers


def conf_bart_shuffle(conf_datadir, tokenizer, batch_size, max_length):
    ds = load_from_disk(conf_datadir)
    ds = ds.shuffle(seed=42)
    # ds = ds.select(range(10))
    ds = ds.train_test_split(test_size=0.1)

    def preprocess_function(
        examples, tokenizer, max_length=1024, abst_length=255, max_target_length=640
    ):
        if "pegasus" in tokenizer.name_or_path:
            token = tokenizers.AddedToken(content="\n", normalized=False)
            tokenizer.add_tokens(list([token]))

        inputs = [f"Input: {x}\n\n" for x in examples["input"]]
        new_line_token = tokenizer.encode("\n", add_special_tokens=False)

        # model_inputs = tokenizer(inputs, truncation=True, max_length=255)
        model_inputs = tokenizer(inputs, truncation=True, max_length=abst_length)

        outputs = []
        for i, (input_text, refs, output) in enumerate(
            zip(model_inputs["input_ids"], examples["references"], examples["output"])
        ):  
            random.shuffle(refs)

            output_org = output
            refs_new = []
            for ref_id, ref in enumerate(refs, 1):
                old_ref_id = re.findall(r"\[\d*\]", ref)[0]
                ref = ref.replace(old_ref_id, f"[{ref_id}]")
                output = output.replace(old_ref_id, f"<<{ref_id}>>")
                refs_new.append(ref)
            refs = refs_new
            output = output.replace("<<", "[")
            output = output.replace(">>", "]")
            outputs.append(output)

            ref_max = math.floor((max_length - abst_length) / len(refs)) - 2 * len(
                new_line_token
            )
            ref_tokens = tokenizer(
                refs, truncation=True, max_length=ref_max, add_special_tokens=False
            )

            ref_token = []
            for ref in ref_tokens["input_ids"]:
                ref_token.extend(ref + new_line_token)
            # input_sample = input_text + ref_token
            input_sample = (
                input_text[:-1] + new_line_token + ref_token + [tokenizer.eos_token_id]
            )
            attention_mask = [1] * len(input_sample)

            model_inputs["input_ids"][i] = input_sample
            model_inputs["attention_mask"][i] = attention_mask

            if len(input_sample) > max_length:
                print(1)

        labels = tokenizer(
            outputs,
            max_length=max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = labels["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels
        if len(model_inputs["input_ids"]) > max_length:
            print(1)
        return model_inputs

    processed_datasets = ds.map(
        lambda x: preprocess_function(x, tokenizer, max_length=max_length),
        batched=True,
        num_proc=1,
        remove_columns=ds["train"].column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["test"]
    test_dataset = processed_datasets["test"]

    # def collate_fn(examples):
    # return tokenizer.pad(examples, padding="longest", return_tensors="pt")
    def collate_fn(examples):
        return tokenizer.pad(
            examples, padding="max_length", max_length=max_length, return_tensors="pt"
        )

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=batch_size,
        pin_memory=True,
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=collate_fn, batch_size=batch_size, pin_memory=True
    )
    test_dataloader = DataLoader(
        test_dataset, collate_fn=collate_fn, batch_size=batch_size, pin_memory=True
    )

    return train_dataloader, eval_dataloader, test_dataloader


def conf_bart_test_shuffle(conf_datadir, tokenizer, batch_size, max_length=1024):
    # if Path(dataset_cache).is_dir() == True:
    ds = load_from_disk(conf_datadir)
    ds = ds.shuffle(seed=42)

    def preprocess_function(
        examples, tokenizer, max_length=1024, abst_length=256, max_target_length=640
    ):
        if "pegasus" in tokenizer.name_or_path:
            token = tokenizers.AddedToken(content="\n", normalized=False)
            tokenizer.add_tokens(list([token]))
        inputs = [f"Input: {x}\n\n" for x in examples["input"]]
        new_line_token = tokenizer.encode("\n", add_special_tokens=False)

        model_inputs = tokenizer(inputs, truncation=True, max_length=abst_length)

        outputs = []
        for i, (input_text, refs, output) in enumerate(
            zip(model_inputs["input_ids"], examples["references"], examples["output"])
        ):
            random.shuffle(refs)

            output_org = output
            refs_new = []
            for ref_id, ref in enumerate(refs, 1):
                old_ref_id = re.findall(r"\[\d*\]", ref)[0]
                ref = ref.replace(old_ref_id, f"[{ref_id}]")
                output = output.replace(old_ref_id, f"<<{ref_id}>>")
                refs_new.append(ref)
            refs = refs_new
            output = output.replace("<<", "[")
            output = output.replace(">>", "]")
            outputs.append(output)

            ref_max = math.floor((max_length - abst_length) / len(refs)) - len(
                new_line_token
            )
            ref_tokens = tokenizer(
                refs, truncation=True, max_length=ref_max, add_special_tokens=False
            )

            ref_token = []
            for ref in ref_tokens["input_ids"]:
                ref_token.extend(ref + new_line_token)
            # input_sample = input_text + ref_token
            input_sample = input_text[:-1] + ref_token + [tokenizer.eos_token_id]
            attention_mask = [1] * len(input_sample)

            # model_inputs['input_ids'][i] = [tokenizer.pad_token_id] * (max_length - len(input_sample)) + sample_input_ids
            # model_inputs['attention_mask'][i] = [0] * (max_length - len(input_sample)) + attention_mask
            # model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])
            # model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])

            model_inputs["input_ids"][i] = input_sample
            model_inputs["attention_mask"][i] = attention_mask

        labels = tokenizer(
            outputs,
            max_length=max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = labels["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels
        return model_inputs

    processed_datasets = ds.map(
        lambda x: preprocess_function(x, tokenizer, max_length=max_length),
        batched=True,
        num_proc=1,
        remove_columns=ds.column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )

    def collate_fn(examples):
        return tokenizer.pad(
            examples, padding="max_length", max_length=max_length, return_tensors="pt"
        )

    test_dataloader = DataLoader(
        processed_datasets,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=batch_size,
        pin_memory=True,
    )

    return test_dataloader, ds["output"], ds["ref_numbers"]




if __name__=='__main__':

    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large", cache_dir="ckpt/bart")
    tokenizer.pad_token = tokenizer.eos_token

    train_dataloader, eval_dataloader, test_dataloader = conf_bart_shuffle("/workdir/hg_dataset/conf23_rw", tokenizer, 4, 1024)

    for data in train_dataloader:
        print(1)
    print(1)
