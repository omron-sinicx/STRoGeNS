from datasets import load_dataset, load_from_disk
import math

from torch.utils.data import DataLoader

def gen_dataset_wrwg(conf_datadir, tokenizer, batch_size, max_length=8192, abst_length=255, max_output_length=640):

    ds = load_from_disk(conf_datadir)
    ds = ds.shuffle(seed=42)
    # ds = ds.select(range(10))
    ds = ds.train_test_split(test_size=0.1)

    train_dataset = ds['train']
    val_dataset = ds['test']

    def process_data_to_model_inputs(batch):
        outputs = tokenizer(batch["related_work"], padding="max_length", truncation=True, max_length=max_output_length)
        inputs = [f"Target paper:\nTitle: {title}\nAbstract: {abst}\n\n" for title, abst in zip(batch["title"], batch['abstract'])]

        new_line_token = tokenizer.encode("\n", add_special_tokens=False)

        model_inputs = tokenizer(inputs, truncation=True, max_length=abst_length)

        for i, (input_text, refs) in enumerate(zip(model_inputs['input_ids'], batch['cited'])):
            ref_max = math.floor((max_length - abst_length) / len(refs)) - 2 * len(new_line_token)
            ref_tokens = tokenizer(refs, truncation=True, max_length=ref_max, add_special_tokens=False)

            ref_token = [] 
            for ref in ref_tokens['input_ids']:
                ref_token.extend(ref + new_line_token)
            # input_sample = input_text + ref_token
            input_sample = input_text[:-1] + new_line_token + ref_token + [tokenizer.eos_token_id]
            attention_mask = [1] * len(input_sample)

            model_inputs['input_ids'][i] = input_sample
            model_inputs['attention_mask'][i] = attention_mask
        model_inputs = tokenizer.pad(model_inputs, padding='max_length', max_length=max_length)
        batch["input_ids"] = model_inputs['input_ids']

        batch["attention_mask"] = model_inputs['attention_mask']

        # create 0 global_attention_mask lists
        batch["global_attention_mask"] = len(batch["input_ids"]) * [[0 for _ in range(len(batch["input_ids"][0]))]]

        # since above lists are references, the following line changes the 0 index for all samples
        batch["global_attention_mask"][0][0] = 1
        batch["labels"] = outputs.input_ids

        # We have to make sure that the PAD token is ignored
        batch["labels"] = [
            [-100 if token == tokenizer.pad_token_id else token for token in labels]
            for labels in batch["labels"]
        ]

        return batch
    
    train_dataset = train_dataset.map(
        process_data_to_model_inputs,
        batched=True,
        batch_size=batch_size,
        remove_columns=["title", "abstract", "related_work", "cited"],
    )

    val_dataset = val_dataset.map(
        process_data_to_model_inputs,
        batched=True,
        batch_size=batch_size,
        remove_columns=["title", "abstract", "related_work", "cited"],
    )

    train_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
    )
    val_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
    )

    def collate_fn(examples):
        return tokenizer.pad(examples, padding="longest", return_tensors="pt")
    
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=batch_size, pin_memory=True)
    eval_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=batch_size, pin_memory=True)
    test_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=batch_size, pin_memory=True)


    return train_dataloader, eval_dataloader



def gen_dataset_wrwg_test(conf_datadir, tokenizer, batch_size, max_length=8192, abst_length=255, max_output_length=640):

    ds = load_from_disk(conf_datadir)
    # ds = ds.select(range(5))

    def process_data_to_model_inputs(batch):
        outputs = tokenizer(batch["related_work"], padding="max_length", truncation=True, max_length=max_output_length)
        inputs = [f"Target paper:\nTitle: {title}\nAbstract: {abst}\n\n" for title, abst in zip(batch["title"], batch['abstract'])]

        new_line_token = tokenizer.encode("\n", add_special_tokens=False)

        model_inputs = tokenizer(inputs, truncation=True, max_length=abst_length)

        for i, (input_text, refs) in enumerate(zip(model_inputs['input_ids'], batch['cited'])):
            ref_max = math.floor((max_length - abst_length) / len(refs)) - 2 * len(new_line_token)
            ref_tokens = tokenizer(refs, truncation=True, max_length=ref_max, add_special_tokens=False)

            ref_token = [] 
            for ref in ref_tokens['input_ids']:
                ref_token.extend(ref + new_line_token)
            input_sample = input_text[:-1] + new_line_token + ref_token + [tokenizer.eos_token_id]
            attention_mask = [1] * len(input_sample)

            model_inputs['input_ids'][i] = input_sample
            model_inputs['attention_mask'][i] = attention_mask
        model_inputs = tokenizer.pad(model_inputs, padding='max_length', max_length=max_length)
        batch["input_ids"] = model_inputs['input_ids']

        batch["attention_mask"] = model_inputs['attention_mask']

        # create 0 global_attention_mask lists
        batch["global_attention_mask"] = len(batch["input_ids"]) * [[0 for _ in range(len(batch["input_ids"][0]))]]

        # since above lists are references, the following line changes the 0 index for all samples
        batch["global_attention_mask"][0][0] = 1
        batch["labels"] = outputs.input_ids

        # We have to make sure that the PAD token is ignored
        batch["labels"] = [
            [-100 if token == tokenizer.pad_token_id else token for token in labels]
            for labels in batch["labels"]
        ]

        return batch
    
    test_dataset = ds.map(process_data_to_model_inputs,batched=True,batch_size=batch_size,remove_columns=["title", "abstract", "related_work", "cited"])

    test_dataset.set_format(type="torch",columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],)
    
    def collate_fn(examples):
        return tokenizer.pad(examples, padding="longest", return_tensors="pt")
    
    test_dataloader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=batch_size, pin_memory=True)

    return test_dataloader, ds['related_work']