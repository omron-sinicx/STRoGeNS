import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy import stats

def calcurate_corr(x, y):
    
    r = stats.spearmanr(x, y) 
    ro = stats.pearsonr(x, y) 
    tau = stats.kendalltau(x, y) 
    return r, ro, tau


def parse_args():
    parser = argparse.ArgumentParser(description="Train data path")
    parser.add_argument("--annot_dir", default="data/annotations/humanevaluation")
    parser.add_argument("--llama2_autoeval", default="outputs/llama2_eval.csv")
    args = parser.parse_args()
    return args

if __name__=="__main__": 
    args = parse_args()
    
    annot_paths = sorted(Path(args.annot_dir).glob("*.xlsx"))
    eval_res = pd.read_csv(args.llama2_autoeval)

    annot_score_total = []
    for annot_path in annot_paths:
        annot = pd.read_excel(str(annot_path), sheet_name=1)
        annot_sort = annot.iloc[:25].sort_values("Unnamed: 0")
        annot_score = annot_sort["Structure score"].values
        annot_score_total.append(annot_score)
    annot_score = np.stack(annot_score_total).mean(0)

    ids = annot_sort.iloc[:25]['Unnamed: 0'].values
    eval_res = eval_res[eval_res['Unnamed: 0'].isin(ids)]

    # f1 ari ari_dot mae mc
    for metric in ['rouge1', 'rouge2', 'rougel', 'f1', 'ari', 'ari_dot', 'mae']:
        
        auto_eval = eval_res[metric].values.astype(np.float32)
        r, ro, tau = calcurate_corr(auto_eval, annot_score)
        print(f"{metric} &{r.statistic * 100:.1f}& {ro.statistic * 100:.1f}& {tau.statistic * 100:.1f}\\\\")
