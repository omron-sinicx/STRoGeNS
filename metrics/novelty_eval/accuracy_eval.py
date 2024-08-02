import argparse
from pathlib import Path
import json

from tqdm import tqdm
import pandas as pd
from datasets import Dataset as hg_Dataset
from openai import OpenAI
import openai
import time

OPENAI_API_KEY='{GPTAPIKEY}'

GPT_LENGTH = {"gpt-3.5-turbo-1106": 10000, "gpt-4-0613": 6000}

GPT_PROMPT = {
    "instruction": "You will be given one related work, title, and abstract written for a computer science paper. Your task is to rate the related work on several metrics. Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.",
    "evaluation_criteria": "Novelty statement (0 or 1) - whether related work contains novelty statement.",
    "generation_step": "1. Read the Related Work Section Thoroughly: Begin by reading the entire related work section carefully. This will give you a complete understanding of the context and how the authors have positioned their work in relation to existing research.\n\n2. Identify the Novelty Statement: Look specifically for a statement or a set of statements where the authors articulate what is new or different about their work compared to the existing literature. This can often be found towards the end of the related work section, but it might also be interspersed throughout the section.\n\n3. Evaluate the Novelty Statement: Presence (0 or 1): Determine if there is a clear statement of novelty. If such a statement exists, score it as '1'. If there is no explicit or implicit statement that outlines what makes the paper's contribution new or unique, score it as '0'.",
    "output_format": "Output only rate with JSON format.\nNovelty statement: {}",
    "system_prompt_format": "{INST}\nEvaluation Criteria:\n\n{E}\n\nEvaluation Steps:\n{S}",
    "user_prompt_format": "Related work: {{D}}\n\nOutput format: {F}"
}

def send_request(client, system_prompt, input_prompt, model_name, max_retries=5):
    retry_wait = 1  # Initial waiting time
    for i in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": input_prompt}
                ]
            )
            time.sleep(1)
            return response
        except openai.RateLimitError as e:
            print(f"{e} Too much requests. Wait {retry_wait} seconds...")
            time.sleep(retry_wait)
            retry_wait *= 2  # increase waiting time
        except openai.InternalServerError as e:
            print(f"{e} Internal error. Wait {retry_wait} ...")
            time.sleep(retry_wait)
            retry_wait *= 2  
    return 1


def load_prompt():
    instruction, eval_criteria, eval_steps, out_form = GPT_PROMPT["instruction"], GPT_PROMPT["evaluation_criteria"], GPT_PROMPT["generation_step"], GPT_PROMPT["output_format"]
    system_prompt = GPT_PROMPT['system_prompt_format'].replace("{INST}", instruction).replace("{E}", eval_criteria).replace("{S}", eval_steps)
    user_prompt_format = GPT_PROMPT['user_prompt_format'].replace("{F}", out_form)
    return system_prompt, user_prompt_format

def evaluation(generated_list, label_list):
    tp = 0
    for pred, gt in zip(generated_list, label_list):
        if pred == gt:
            tp += 1
    acc = tp / len(generated_list)
    print(acc)

def main(args):
    Path(args.save_path).mkdir(parents=True, exist_ok=True)

    # load dataset
    dataset = pd.read_csv(args.data_dir)
    data_list = dataset['text'].values
    label_list = dataset['label'].values
    
    # load prompt
    system_prompt, user_prompt_format = load_prompt()

    # define openai api
    client = OpenAI(api_key=OPENAI_API_KEY, max_retries=5)

    # Start from intermediate result
    if args.temp_dir is not None:
        df = pd.read_csv(args.temp_dir)
        dataset = dataset.select(range(len(df['gt']), len(dataset)))
        generated_list = df['pred'].values.tolist()
    else:    
        generated_list = []

    # Start loop
    for idx, data in enumerate(tqdm(data_list)): 
        input_prompt = user_prompt_format.replace("{D}", data)

        # Request to API
        try:
            response = send_request(client, system_prompt, input_prompt, args.model, max_retries=5)
            if response == 1:
                df = pd.DataFrame({"pred": generated_list})
                df.to_csv(f"{args.save_path}/{args.model}_temporal.csv")
        except:
            # save temporal result
            df = pd.DataFrame({"pred": generated_list})
            df.to_csv(f"{args.save_path}/{args.model}_temporal.csv")
            
            import sys 
            sys.exit()
        
        generated_list.append(response.choices[0].message.content)
        df = pd.DataFrame({"pred": generated_list})
        df.to_csv(f"{args.save_path}/{args.model}_temporal.csv")


    df = pd.DataFrame({"pred": generated_list, "gt": label_list})
    df.to_csv(f"{args.save_path}/{args.model}.csv")

    pred_list = []
    for pred in df['pred'].values:
        pred_list.append(list(json.loads(pred).values())[0])
    
    evaluation(pred_list, label_list)


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Train data path")
    # parser.add_argument("--model", default="gpt-3.5-turbo-1106", type=str)
    parser.add_argument("--model", default="gpt-4-0613", type=str)
    parser.add_argument("--save_path", default="outputs/nove_eval_by_geval", type=str)
    parser.add_argument("--data_dir", default="data/annotations/novelty_annotation.csv", type=str)
    parser.add_argument("--temp_dir", default=None, type=str)
    args = parser.parse_args()
    return args

if __name__=='__main__':
    args = parse_args()

    main(args)