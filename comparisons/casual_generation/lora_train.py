import argparse
from pathlib import Path
import pickle

from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    get_linear_schedule_with_warmup,
    set_seed,
)
from peft import LoraConfig, TaskType, get_peft_model, PeftModel, PeftConfig
from accelerate import Accelerator


from dataloader import conf_dataloader

import _init_paths
from metrics import rouge_reference_overall


def evaluation(model, eval_dataloader, tokenizer):
    model.eval()

    total_loss = 0
    for step, batch in enumerate(tqdm(eval_dataloader)):
        batch = {k: v for k, v in batch.items() if k != "gt"}
        with torch.no_grad():

            outputs = model(**batch)
            loss = outputs.loss.cpu().detach()
        total_loss += loss.float()
    eval_epoch_loss = total_loss / len(eval_dataloader)

    return eval_epoch_loss

def save_model(ckpt_path, accelerator, model):
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        ckpt_path,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=accelerator.get_state_dict(model),
    )

def train(model, train_dataloader, epoch, accelerator, optimizer, lr_scheduler):
    model.train()
    total_loss = 0
    with tqdm(train_dataloader) as pbar:
        for step, batch in enumerate(tqdm(train_dataloader)):
            outputs = model(**batch)
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


def main(args):
    accelerator = Accelerator(log_with=args.log_with)
    if args.log_with == "wandb":
        accelerator.init_trackers(project_name="llm_finetuning")
        try:
            accelerator.trackers[0].run.name = args.run_name
        except:
            pass

    set_seed(args.seed)
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)

    # creating model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
    tokenizer.pad_token = tokenizer.eos_token

    # dataset
    train_dataloader, eval_dataloader, test_dataloader = conf_dataloader(
        args.data_dir, tokenizer, args.batch_size, args.max_token_length
    )
    accelerator.wait_for_everyone()

    # creating model
    if args.resume is not None:
        max_memory = {0: "1GIB", 1: "1GIB", 2: "2GIB", 3: "10GIB", "cpu": "30GB"}
        model = AutoModelForCausalLM.from_pretrained(
            args.model, 
            cache_dir=args.cache_dir, 
            use_flash_attention_2=True # Comment out according to GPU
        )
        model.enable_input_require_grads()
        model = PeftModel.from_pretrained(model, args.resume, max_memory=max_memory, is_trainable=True)

        model.gradient_checkpointing_enable()
        model.print_trainable_parameters()

    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            cache_dir=args.cache_dir,
            use_flash_attention_2=True # Comment out according to GPU
        )
        model.enable_input_require_grads()

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.peft_r,
            lora_alpha=args.peft_alpha,
            lora_dropout=args.peft_dropout,
        )
        model = get_peft_model(model, peft_config)
        model.gradient_checkpointing_enable()

        model.print_trainable_parameters()
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
        model, train_dataloader, eval_dataloader, test_dataloader, optimizer, lr_scheduler
    )

    eval_loss_min = 1e5
    for epoch in range(args.num_epochs):
        train_loss = train(model, train_dataloader, epoch, accelerator, optimizer, lr_scheduler)
        print(train_loss)

        eval_epoch_loss = evaluation(model, eval_dataloader, tokenizer)
        if eval_loss_min > eval_epoch_loss:
            model.save_pretrained(f"{args.ckpt_dir}/{args.ckpt_best}")
            accelerator.save_model(model, f"{args.ckpt_dir}/{epoch}")
            save_model(f"{args.ckpt_dir}/{epoch}", accelerator, model)

        save_model(f"{args.ckpt_dir}/{epoch}", accelerator, model)
        accelerator.save_state(f"{args.ckpt_dir}/{epoch}")
        accelerator.log({"train_loss": train_loss, "eval_loss": eval_epoch_loss}, step=epoch)
    accelerator.end_training()
    
    save_model(f"{args.ckpt_dir}/{epoch}", accelerator, model)


def parse_args():
    """
    Parse input arguments
    """
    # parser.add_argument("--model", default="HuggingFaceH4/zephyr-7b-beta", type=str)
    # parser.add_argument("--cache_dir", default="ckpt/zephyr", type=str)

    parser = argparse.ArgumentParser(description="Train data path")
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1", type=str)
    parser.add_argument("--cache_dir", default="ckpt/mistral", type=str)

    parser.add_argument("--data_dir", default="hg_dataset/conf_rw", type=str)
    parser.add_argument("--ckpt_dir", default="./ckpt/finetuningv2/mistral_conf", type=str)

    parser.add_argument("--ckpt_inter", default="inter", type=str)
    parser.add_argument("--ckpt_best", default="best", type=str)
    parser.add_argument("--ckpt_final", default="final", type=str)

    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--lr", default=3e-5, type=float)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--num_epochs", default=20, type=int)
    parser.add_argument("--max_token_length", default=8192, type=int)
    parser.add_argument("--resume", default=None, type=str)

    # wandb
    # parser.add_argument("--log_with", default="wandb", type=str)
    parser.add_argument("--log_with", default=None, type=str)
    parser.add_argument("--run_name", default="single_sample", type=str)

    # peft options
    parser.add_argument("--peft_r", default=64, type=int)
    parser.add_argument("--peft_alpha", default=32, type=int)
    parser.add_argument("--peft_dropout", default=0.1, type=float)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
