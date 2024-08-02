import argparse
from pathlib import Path
import pickle

from tqdm import tqdm
from summa import summarizer
from datasets import Dataset as hg_Dataset
from datasets import load_from_disk

import pandas as pd 

import _init_paths
from metrics import rouge_reference_overall

def main(args):
    # dataload
    dataset = load_from_disk(args.data_dir)
    dataset = dataset.select(range(10))

    gt_list = []
    pred_list = []
    r1_list = []
    r2_list = []
    rl_list = []
    for data in tqdm(dataset):
        title = data['title']
        abst = data['abstract']

        input_text = f"Target paper:\nTitle: {title}\nAbstract: {abst}\n\n"
        for ref_text in data['cited']:
            input_text += ref_text
        summary = summarizer.summarize(input_text)
        pred_list.append(summary)

        gt_list.append(data['related_work'])

        score = rouge_reference_overall([data['related_work']], [summary])
        r1_list.append(score['rouge-1-f'])
        r2_list.append(score['rouge-2-f'])
        rl_list.append(score['rouge-l-f'])

    df = pd.DataFrame({"pred": pred_list, "reference": gt_list, "r1_score": r1_list, "r2_score": r2_list, "rl_score": rl_list})
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(f"{args.output_dir}/text_rank.csv")

    print(df['r1_score'].mean(), df['r2_score'].mean(), df['rl_score'].mean())


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Train data path")
    parser.add_argument("--data_dir", default="/scratch/acc12552uq/llm_code_organize/intern_2023_KazuyaNishimura/hg_dataset_new/conf23_rw", type=str)
    parser.add_argument("--output_dir", default="outputs/text_rank", type=str)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()
    return args

if __name__=='__main__':
    args = parse_args()
    main(args)