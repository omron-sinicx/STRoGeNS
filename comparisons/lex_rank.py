import argparse
from pathlib import Path

from lexrank import LexRank
from lexrank.mappings.stopwords import STOPWORDS
from datasets import Dataset as hg_Dataset
from datasets import load_from_disk
from tqdm import tqdm
import pandas as pd

import _init_paths
from metrics import rouge_reference_overall


def main(args):
    # dataload
    dataset = load_from_disk(args.data_dir)
    # dataset = dataset.select(range(10))

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

        lxr = LexRank([input_text.split("\n\n")], stopwords=STOPWORDS['en'])

        input_sentences = input_text.replace("\n\n", "").split(".") 

        summary = lxr.get_summary(input_sentences, summary_size=20, threshold=.1)
        summary = ''.join(summary)

        pred_list.append(summary)

        gt_list.append(data['related_work'])

        score = rouge_reference_overall([data['related_work']], [summary])
        r1_list.append(score['rouge-1-f'])
        r2_list.append(score['rouge-2-f'])
        rl_list.append(score['rouge-l-f'])

    df = pd.DataFrame({"pred": pred_list, "reference": gt_list, "r1_score": r1_list, "r2_score": r2_list, "rl_score": rl_list})
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(f"{args.output_dir}/lex_rank.csv")

    print(df['r1_score'].mean(), df['r2_score'].mean(), df['rl_score'].mean())



def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Train data path")
    parser.add_argument("--data_dir", default="data/STRoGeNS-arXiv22/hg_format", type=str)
    parser.add_argument("--output_dir", default="outputs/lex_rank", type=str)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()
    return args


if __name__=='__main__':
    args = parse_args()
    main(args)