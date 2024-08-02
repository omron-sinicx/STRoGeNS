import argparse
from pathlib import Path
import pickle

from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    default_data_collator,
    get_linear_schedule_with_warmup,
    set_seed,
)
import tokenizers
from peft import LoraConfig, TaskType, get_peft_model, PeftModel, PeftConfig
from accelerate import Accelerator

from led_dataloader import gen_dataset_wrwg

import _init_paths

from metrics import rouge_reference_overall


def evaluation(model, eval_dataloader, tokenizer):
    model.eval()

    total_loss = 0
    for step, batch in enumerate(tqdm(eval_dataloader)):
        batch = {k: v for k, v in batch.items() if k != "gt"}
        with torch.no_grad():
            outputs = model(batch['input_ids'], attention_mask=batch['attention_mask'],global_attention_mask=batch['global_attention_mask'], labels=batch['labels'])
        loss = outputs.loss
        total_loss += loss.detach().float()
    eval_epoch_loss = total_loss / len(eval_dataloader)

    return eval_epoch_loss


def train(model, train_dataloader, epoch, accelerator, optimizer, lr_scheduler):
    model.train()
    total_loss = 0

    with tqdm(train_dataloader) as pbar:
        for step, batch in enumerate(pbar):
            outputs = model(batch['input_ids'], attention_mask=batch['attention_mask'],global_attention_mask=batch['global_attention_mask'], labels=batch['labels']) 
            loss = outputs.loss
            total_loss += loss.detach().float()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            pbar.set_description(f'loss: {loss.detach().float():.4f}')
        train_epoch_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(train_epoch_loss)
        accelerator.print(f"{epoch=}: {train_ppl=} {train_epoch_loss=}")
    return train_epoch_loss

def save_model(ckpt_path, accelerator, model):
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        ckpt_path,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=accelerator.get_state_dict(model),
    )

def main(args):
    accelerator = Accelerator(log_with=args.log_with)
    if args.log_with == "wandb":
        accelerator.init_trackers(project_name="llm_finetuning")
        accelerator.trackers[0].run.name = args.run_name

    set_seed(args.seed)
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # train_dataset, val_dataset= gen_dataset(tokenizer)
    train_dataloader, eval_dataloader = gen_dataset_wrwg(args.data_dir, tokenizer, args.batch_size)

    if args.resume is not None:
        model = AutoModelForSeq2SeqLM.from_pretrained(f"{args.resume}/{args.ckpt_best}", cache_dir=args.cache_dir)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model, cache_dir=args.cache_dir, gradient_checkpointing=True)
    # set generate hyperparameters
    model.config.num_beams = 2
    model.config.max_length = 640
    model.config.min_length = 100
    model.config.length_penalty = 2.0
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    
    accelerator.wait_for_everyone()

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # lr scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * args.num_epochs),
    )

    # accelerate preparement
    accelerator.wait_for_everyone()
    model, train_dataloader, eval_dataloader, test_dataloader, optimizer, lr_scheduler = accelerator.prepare(
        model, train_dataloader, eval_dataloader, eval_dataloader, optimizer, lr_scheduler
    )

    eval_loss_min = 1e5
    for epoch in range(args.num_epochs):
        train_loss = train(model, train_dataloader, epoch, accelerator, optimizer, lr_scheduler)
        
        eval_epoch_loss = evaluation(model, eval_dataloader, tokenizer)
        if eval_loss_min > eval_epoch_loss:
            save_model(f"{args.ckpt_dir}/{args.ckpt_best}", accelerator, model)

        save_model(f"{args.ckpt_dir}/{args.ckpt_inter}", accelerator, model)
        accelerator.log({"train_loss": train_loss, "eval_loss": eval_epoch_loss}, step=epoch)
    accelerator.end_training()

    save_model(f"{args.ckpt_dir}/{args.ckpt_final}", accelerator, model)

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Train data path")
    parser.add_argument("--model", default="allenai/led-large-16384", type=str)
    parser.add_argument("--cache_dir", default="ckpt/led", type=str)
    parser.add_argument("--data_dir", default="./hg_dataset/conf_rw", type=str)
    parser.add_argument("--ckpt_dir", default="./ckpt/finetuning/led_conf", type=str)
    
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--lr", default=3e-5, type=float)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--num_epochs", default=20, type=int)
    parser.add_argument("--max_token_length", default=8192, type=int)
    parser.add_argument("--resume", default=None, type=str)
    
    parser.add_argument("--ckpt_inter", default="inter", type=str)
    parser.add_argument("--ckpt_best", default="best", type=str)
    parser.add_argument("--ckpt_final", default="final", type=str)
    

    # wandb
    # parser.add_argument("--log_with", default="wandb", type=str)
    parser.add_argument("--log_with", default=None, type=str)
    parser.add_argument("--run_name", default="bert_training", type=str)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
