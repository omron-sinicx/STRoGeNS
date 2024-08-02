import argparse

import pandas as pd
import torch
from torch.utils.data import DataLoader

from datasets import load_dataset, load_metric
from transformers import LEDTokenizer, LEDForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import EarlyStoppingCallback

from datasets import load_dataset, load_metric

from led_dataloader import gen_dataset_wrwg, gen_dataset_wrwg_test
import _init_paths
from metrics import rouge_reference_overall, cluster_eval

import evaluate

rouge = evaluate.load("rouge")


def gen_dataset(tokenizer, batch_size):
    max_input_length = 8192
    max_output_length = 640

    train_dataset = load_dataset("scientific_papers", "pubmed", split="train")
    val_dataset = load_dataset("scientific_papers", "pubmed", split="validation")

    train_dataset = train_dataset.select(range(250))
    val_dataset = val_dataset.select(range(25))

    def process_data_to_model_inputs(batch):
        # tokenize the inputs and labels
        inputs = tokenizer(
            batch["article"],
            padding="max_length",
            truncation=True,
            max_length=max_input_length,
        )
        outputs = tokenizer(
            batch["abstract"],
            padding="max_length",
            truncation=True,
            max_length=max_output_length,
        )

        batch["input_ids"] = inputs.input_ids
        batch["attention_mask"] = inputs.attention_mask

        # create 0 global_attention_mask lists
        batch["global_attention_mask"] = len(batch["input_ids"]) * [[0 for _ in range(len(batch["input_ids"][0]))]]

        # since above lists are references, the following line changes the 0 index for all samples
        batch["global_attention_mask"][0][0] = 1
        batch["labels"] = outputs.input_ids

        # We have to make sure that the PAD token is ignored
        batch["labels"] = [
            [-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]
        ]

        return batch

    train_dataset = train_dataset.map(
        process_data_to_model_inputs,
        batched=True,
        # batch_size=batch_size,
        remove_columns=["article", "abstract", "section_names"],
    )

    val_dataset = val_dataset.map(
        process_data_to_model_inputs,
        batched=True,
        batch_size=batch_size,
        remove_columns=["article", "abstract", "section_names"],
    )

    train_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
    )
    val_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
    )

    return train_dataset, val_dataset


def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge1", "rouge2"])

    return {
        "rouge1_fmeasure": round(rouge_output["rouge1"], 4),
        "rouge2_fmeasure": round(rouge_output["rouge2"], 4),
    }


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Train data path")
    parser.add_argument("--model", default="allenai/led-large-16384", type=str)
    parser.add_argument("--cache_dir", default="ckpt/led", type=str)

    parser.add_argument("--data_dir", default="/workdir/hg_dataset/conf23_rw", type=str)
    parser.add_argument("--ckpt_dir", default="/workdir/ckpt/finetuning/led_conf", type=str)
    parser.add_argument("--save_path", default="output/cond_pred/bart", type=str)
    parser.add_argument("--run_name", default="output/cond_pred/bart", type=str)

    parser.add_argument("--resume", default=None, type=str)

    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--lr", default=3e-5, type=float)
    parser.add_argument("--num_epochs", default=20, type=int)
    parser.add_argument("--max_token_length", default=8192, type=int)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # train_dataset, val_dataset= gen_dataset(tokenizer)
    train_dataset, val_dataset = gen_dataset_wrwg(args.data_dir, tokenizer, args.batch_size)

    if args.resume is not None:
        led = AutoModelForSeq2SeqLM.from_pretrained(args.resume, cache_dir=args.cache_dir, gradient_checkpointing=True)
    else:
        led = AutoModelForSeq2SeqLM.from_pretrained(args.model, cache_dir=args.cache_dir, gradient_checkpointing=True)
    # set generate hyperparameters
    led.config.num_beams = 2
    led.config.max_length = 640
    led.config.min_length = 100
    led.config.length_penalty = 2.0
    led.config.early_stopping = True
    led.config.no_repeat_ngram_size = 3

    # # enable fp16 apex training
    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        fp16=True,
        output_dir=args.ckpt_dir,
        logging_steps=5,
        # eval_steps=10,
        eval_steps=5,
        # save_steps=10,
        save_total_limit=2,
        gradient_accumulation_steps=1,
        learning_rate=args.lr,
        num_train_epochs=args.num_epochs,
        # report_to="wandb",
        report_to="none",
        run_name=args.run_name,
        load_best_model_at_end=True,
    )

    trainer = Seq2SeqTrainer(
        model=led,
        tokenizer=tokenizer,
        args=training_args,
        accelerator="ddp",
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=8)],
    )

    trainer.train()

    trainer.save_model(f"{args.ckpt_dir}/best")

    pubmed_test = load_dataset("scientific_papers", "pubmed", ignore_verifications=True, split="test")

    # load tokenizer
    tokenizer = LEDTokenizer.from_pretrained(f"{args.ckpt_dir}/best")
    model = LEDForConditionalGeneration.from_pretrained(f"{args.ckpt_dir}/best").to("cuda").half()
    # tokenizer = LEDTokenizer.from_pretrained("patrickvonplaten/led-large-16384-pubmed")
    # model = LEDForConditionalGeneration.from_pretrained("patrickvonplaten/led-large-16384-pubmed").to("cuda").half()

    # load pubmed
    test_dataset, gt_list, ref_list = gen_dataset_wrwg_test("/workdir/hg_dataset/conf23_rw", tokenizer, args.batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, pin_memory=True)

    generated_list = []
    for batch in test_dataloader:
        # inputs_dict = tokenizer(batch["article"], padding="max_length", max_length=8192, return_tensors="pt", truncation=True)
        input_ids = batch["input_ids"].to("cuda")
        attention_mask = batch["attention_mask"].to("cuda")
        global_attention_mask = batch["global_attention_mask"].to("cuda")

        # inputs_dict = tokenizer(batch["article"], padding="max_length", max_length=8192, return_tensors="pt", truncation=True)
        # input_ids = inputs_dict.input_ids.to("cuda")
        # attention_mask = inputs_dict.attention_mask.to("cuda")
        # global_attention_mask = torch.zeros_like(attention_mask)
        # # put global attention on  token
        # global_attention_mask[:, 0] = 1

        predicted_abstract_ids = model.generate(
            input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask
        )
        output_texts = tokenizer.batch_decode(predicted_abstract_ids, skip_special_tokens=True)

        generated_list.extend(output_texts)

    df = pd.DataFrame({"pred": generated_list, "gt": gt_list, "ref": ref_list})
    df.to_csv(args.save_path)

    df = pd.read_csv(args.save_path)

    generated_list = df["pred"].values
    ref_list = df["ref"].values
    gt_list = df["gt"].values
    rouge_scores = rouge_reference_overall(gt_list, generated_list)
    print("rouge 1, rouge 2, rouge l")
    print(rouge_scores["rouge-1-f"], rouge_scores["rouge-2-f"], rouge_scores["rouge-l-f"])
    num_non_cited, cluster_score, num_paragraph = cluster_eval(gt_list, ref_list, generated_list)
    print("non_cited, ARI, num_para")
    print(num_non_cited, cluster_score, num_paragraph)

    print(
        f"{rouge_scores['rouge-1-f']}& {rouge_scores['rouge-2-f']}& {rouge_scores['rouge-l-f']}& {num_non_cited}& {cluster_score}& {num_paragraph}"
    )

    import numpy as np

    np.save(
        f"{args.save_path}_metrics",
        [
            rouge_scores["rouge-1-f"],
            rouge_scores["rouge-2-f"],
            rouge_scores["rouge-l-f"],
            num_non_cited,
            cluster_score,
            num_paragraph,
        ],
    )
