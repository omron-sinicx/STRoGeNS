import argparse
from pathlib import Path
import json

from tqdm import tqdm
import pandas as pd
from datasets import Dataset as hg_Dataset
from openai import OpenAI
import openai
import time
import tiktoken
from datasets import load_from_disk

import _init_paths
from metrics import rouge_reference_overall, cluster_eval
from GPT_estimation.dataset import convertdict

OPENAI_API_KEY='{GPTAPIKEY}'

GPT_LENGTH = {"gpt-3.5-turbo-1106": 10000, "gpt-4-0613": 6000}

def send_request(client, system_prompt, input_prompt, model_name, max_retries=5):
    retry_wait = 1  # Initial waiting time
    for i in range(max_retries):
        try:
            time.sleep(0.5)
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": input_prompt}
                ]
            )
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


def evaluation(df):
    generated_list = df['pred'].values
    ref_list = df['ref'].values
    gt_list = df['gt'].values
    rouge_scores = rouge_reference_overall(gt_list, generated_list)
    print("rouge 1, rouge 2, rouge l")
    print(rouge_scores['rouge-1-f'], rouge_scores['rouge-2-f'], rouge_scores['rouge-l-f'])
    num_non_cited, cluster_score, num_paragraph = cluster_eval(gt_list, ref_list, generated_list)
    print("non_cited, ARI, num_para")
    print(num_non_cited, cluster_score, num_paragraph)


def load_prompt(prompt_config):
    with open(prompt_config, "r") as f:
        prompt = json.load(f)
    instruction, steps, output_format = prompt["instruction"], prompt["generation_step"], prompt["output_format"]
    system_prompt = prompt['system_prompt_format'].replace("{INST}", instruction).replace("{Step}", steps)
    user_prompt_format = prompt['user_prompt_format'].replace("{Out_Form}", output_format)
    return system_prompt, user_prompt_format

def main(args):
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # load dataset
    dataset = load_from_disk(args.data_dir)
    
    # load prompt
    system_prompt, user_prompt_format = load_prompt(args.prompt_config)
    prompt_style = Path(args.prompt_config).stem.split("_")[1]

    # define openai api
    client = OpenAI(api_key=OPENAI_API_KEY, max_retries=5)

    # Start from intermediate result
    if args.temp_dir is not None:
        df = pd.read_csv(args.temp_dir)
        dataset = dataset.select(range(len(df['gt']), len(dataset)))
        generated_list = df['pred'].values.tolist()
        ref_list = df['ref'].values.tolist()
        gt_list = df['gt'].values.tolist()
    else:    
        generated_list = []
        gt_list = []
        ref_list = []

    # Start loop
    for idx, data in enumerate(tqdm(dataset)): 
        input_prompt = user_prompt_format.replace("{D}", data['input'])
        gt = data['output']
        ref = data['references']

        # Request to API
        try:
            response = send_request(client, system_prompt, input_prompt, args.model, max_retries=5)
            if response == 1:
                df = pd.DataFrame({"pred": generated_list, "gt": gt_list, "ref": ref_list})
                df.to_csv(f"{args.output_dir}/{prompt_style}_{args.model}_temporal_save.csv")
        except:
            # save temporal result
            df = pd.DataFrame({"pred": generated_list, "gt": gt_list, "ref": ref_list})
            df.to_csv(f"{args.output_dir}/{prompt_style}_{args.model}_temporal_save.csv")
            
            import sys 
            sys.exit()
        
        generated_list.append(response.choices[0].message.content)
        gt_list.append(gt)
        ref_list.append(ref)

        df = pd.DataFrame({"pred": generated_list, "gt": gt_list, "ref": ref_list})
        df.to_csv(f"{args.output_dir}/{prompt_style}_{args.model}_temporal_save.csv")

    df = pd.DataFrame({"pred": generated_list, "gt": gt_list, "ref": ref_list})
    df.to_csv(f"{args.output_dir}/{prompt_style}_{args.model}.csv")

    df = pd.read_csv(f"{args.output_dir}/{prompt_style}_{args.model}.csv")

    evaluation(df)


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Train data path")
    # parser.add_argument("--model", default="gpt-3.5-turbo-1106", type=str)
    parser.add_argument("--model", default="gpt-4-0613", type=str)
    parser.add_argument("--output_dir", default="outputs/GPT_preds", type=str)
    parser.add_argument("--prompt_config", default="comparisons/GPT_estimation/prompts", type=str)
    parser.add_argument("--data_dir", default="conf22", type=str)

    parser.add_argument("--temp_dir", default=None, type=str)
    args = parser.parse_args()
    return args

if __name__=='__main__':
    args = parse_args()
    main(args)