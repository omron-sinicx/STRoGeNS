import re
import pandas as pd
import numpy as np
from sklearn.metrics import adjusted_rand_score, pair_confusion_matrix

def find_digit_citation(paragraph):
    cites = re.findall(r"\[\d+(?:, \d+)*\]", paragraph)

    # {"original cite marker" : "proceced"}
    digit_cite_list = []
    for org_cite in cites:
        # remove []
        cite = org_cite[1:-1]
        if "," in cite:
            cite = cite.replace(" ", "").split(",")
            norm_cite = ""
            for c in cite:
                norm_cite += f"[{c}]"

            paragraph = paragraph.replace(org_cite, norm_cite)
        else:
            cite = [cite]
        digit_cite_list.extend(cite)
    return paragraph, digit_cite_list

def text2label(gt):
    paragraph = gt.split("\n\n")

    ref_name_list = []
    cluster_id_list = []
    for cluster_id, para in enumerate(paragraph):
        para, digit_list = find_digit_citation(para)

        for digit_id in set(digit_list):
            ref_name_list.append(digit_id)
            cluster_id_list.append(cluster_id)

    return pd.DataFrame({"ref id": ref_name_list, "clu id": cluster_id_list})


def adjusted_rand_score_dot(labels_true, labels_pred, UM=0):
    (tn, fp), (fn, tp) = pair_confusion_matrix(labels_true, labels_pred)
    # convert to Python integer types, to avoid overflow or underflow
    tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)

    # Special cases: empty data or full agreement
    if fn == 0 and fp == 0:
        return 1.0
    
    M = sum([tn, fp, fn, tp])
    norm_tn, norm_fp, norm_fn, norm_tp  = tn/ M, fp/M, fn/M, tp/ M
    # RI - Expected / max(RI) - Expected
    return (((tp + tn) / (M + UM)) - ((norm_tp + norm_fn) * (norm_tp + norm_fp) + (norm_fn + norm_tn) * (norm_fp + norm_tn))) /  (((norm_tp + norm_fn) * (norm_fn + norm_tn) + (norm_tp + norm_fp) * (norm_fp + norm_tn)))


def ari_calc_greedy(gt_df, pred_df):
    # extract single digit 
    for idx, ref_id in gt_df['ref id'].items():
        mat_cand_pred = pred_df[pred_df['ref id'] == ref_id]

        if len(mat_cand_pred) == 1:
            pred_cluster_id = mat_cand_pred['clu id'].values[0]
        else:        
            gt_clu_id = gt_df.loc[idx]['clu id']
            # gather references in the cluster
            ref_in_gt_cand_clu = gt_df[gt_df['clu id'] == gt_clu_id]['ref id'].values

            select_df_and_score = []
            # calculate score for 
            for cand_idx, pred_cand_clu_id in mat_cand_pred['clu id'].items():
                ref_in_pred_cand_clu = pred_df[pred_df['clu id'] == pred_cand_clu_id]['ref id'].values

                match = set(ref_in_gt_cand_clu).intersection(ref_in_pred_cand_clu)
                num_candidates = set(ref_in_gt_cand_clu).union(ref_in_pred_cand_clu)
                score = len(match) / len(num_candidates)
                select_df_and_score.append([pred_cand_clu_id, score])
            select_df_and_score = np.array(select_df_and_score)
            sim_sample_id = select_df_and_score[:, 1].argmax()
            pred_cluster_id = select_df_and_score[sim_sample_id][0]

        gt_df.loc[idx, 'clu id gen'] = pred_cluster_id
    return adjusted_rand_score(gt_df['clu id'].values, gt_df['clu id gen'].values)


def ari_calc_match(gt_df, pred_df):

    unmatched_pred = pred_df.copy()
    # id_list = set(gt_df['ref id'].values)
    # extract single digit 
    for ref_id in np.unique(gt_df['ref id'].values):
        selected_gt_idx = gt_df[gt_df['ref id'] == ref_id]
        
        # extract candidate pragraph from pred 
        mat_cand_pred = unmatched_pred[unmatched_pred['ref id'] == ref_id]

        if len(selected_gt_idx) == 1 and len(mat_cand_pred) == 1:
            gt_df.loc[selected_gt_idx.index[0], 'clu id gen'] = mat_cand_pred['clu id'].values[0]
            unmatched_pred = unmatched_pred.drop(mat_cand_pred.index[0])
            continue

        # calculate mathing score for each candidate
        select_df_and_score = []    
        for gt_idx, gt_cand_clu_id in selected_gt_idx['clu id'].items():
            ref_in_gt_cand_clu = gt_df[gt_df['clu id'] == gt_cand_clu_id]['ref id'].values

            # calculate score for 
            for cand_idx, pred_cand_clu_id in mat_cand_pred['clu id'].items():
                ref_in_pred_cand_clu = pred_df[pred_df['clu id'] == pred_cand_clu_id]['ref id'].values

                match = set(ref_in_gt_cand_clu).intersection(ref_in_pred_cand_clu)
                num_candidates = set(ref_in_gt_cand_clu).union(ref_in_pred_cand_clu)
                score = len(match) / len(num_candidates)
                select_df_and_score.append([gt_cand_clu_id, gt_idx, pred_cand_clu_id, cand_idx, score])

        # gt_cluster_id, gt_idx, pred_cluster_id, pred_idx, score
        select_df_and_score = np.array(select_df_and_score)

        # select most matched sample
        while len(select_df_and_score) > 0:
            match_idx = select_df_and_score[:, -1].argmax()
            gt_df.loc[select_df_and_score[match_idx][1], 'clu id gen'] = select_df_and_score[match_idx][2]
            unmatched_pred = unmatched_pred.drop(index=select_df_and_score[match_idx][3])

            # remove associated gt_samples and pred_samples
            select_df_and_score = select_df_and_score[(select_df_and_score[:, 1] != select_df_and_score[match_idx][1]) & (select_df_and_score[:, 3] != select_df_and_score[match_idx][3])]
        
    # Unmatched_samples =
    cluster_score = adjusted_rand_score_dot(gt_df[gt_df['clu id gen'].notna()][['clu id', 'clu id gen']].values[:, 0], gt_df[gt_df['clu id gen'].notna()][['clu id', 'clu id gen']].values[:, 1])

    unmatched_pred = sum(gt_df['clu id gen'].isna()) + len(unmatched_pred)
    cluster_score_dot = adjusted_rand_score_dot(gt_df[gt_df['clu id gen'].notna()][['clu id', 'clu id gen']].values[:, 0], gt_df[gt_df['clu id gen'].notna()][['clu id', 'clu id gen']].values[:, 1], unmatched_pred)
    return cluster_score, cluster_score_dot


def f1_calc(gt_df, pred_df):
    # calcurate f1 score for each paragraph
    f14paragraphs = []
    for clu_id in gt_df['clu id'].unique():
        cluster_refs = gt_df[gt_df['clu id'] == clu_id]['ref id'].values

        f1_list = []
        for pred_clu_id in pred_df['clu id'].unique():
            pred_cluster_refs = pred_df[pred_df['clu id'] == pred_clu_id]['ref id'].values
            tp = len(set(cluster_refs).intersection(pred_cluster_refs))
            fn = len(set(cluster_refs).difference(pred_cluster_refs))
            fp = len(set(pred_cluster_refs).difference(cluster_refs))

            recall = tp / (tp + fn)
            precision = tp / (tp + fp)

            f1 = 2 * recall * precision / (recall + precision + 1e-10)
            f1_list.append(f1)

        f1_max = max(f1_list)
        if np.isnan(f1_max):
            print(1)
        f14paragraphs.append(f1_max)
    return np.array(f14paragraphs).mean()

def calc_miss_citation_rate(gt_df, pred_df):
    mc_idx = set(gt_df['ref id'].values) - set(pred_df['ref id'].values)
    mc_rate = len(mc_idx) / len(set(gt_df['ref id'].values))

    return mc_rate, gt_df[~gt_df['ref id'].isin(mc_idx)]

def paragraph_eval(gt, generated):
    # MAE para
    mae_para = np.abs(len(generated.split("\n\n")) - len(gt.split("\n\n")))

    gt_df = text2label(gt)
    pred_df = text2label(generated)

    gt_df['clu id gen'] = np.nan

    # calculate miss citation rate and remove miss citation
    mc_rate, gt_df = calc_miss_citation_rate(gt_df, pred_df)

    if len(gt_df) > 1:
        f1 = f1_calc(gt_df, pred_df)
        # calculate miss citation rate and remove miss citation
        ari_greedy = ari_calc_greedy(gt_df.copy(), pred_df.copy())
        ari_mat, ari_dot = ari_calc_match(gt_df.copy(), pred_df.copy())
    else:
        f1 = 0
        ari_greedy = 0
        ari_mat = 0
        ari_dot = 0
    
    return mae_para, mc_rate, f1, ari_greedy, ari_mat, ari_dot




if __name__=="__main__":
    from pathlib import Path
    from datasets import Dataset as hg_Dataset
    from datasets import load_from_disk
    from tqdm import tqdm

    ds = load_from_disk("/Users/nishimurakazuya/relatework_dataset/hg_dataset_v2/conf23_rw_v2")
    gt = ds["output"][2]
    ref = ds["references"][2]
    # print(gt)
    generated = "Model Compression and Fairness in NLP. [1] [2] [3] [4] The intersection of model compression techniques and their impact on fairness in NLP systems is a burgeoning area of research. [1] demonstrated how pruning and model\n\n [1], [2], [5], [6] ,[7]distillation can be integrated to train sparse pre-trained Transformer language models, which maintain their efficiency in various NLP tasks with minimal accuracy loss. This aligns with our focus on model compression methods, highlighting their potential efficiency benefits. Field et al. (2021) provided a critical survey on race and racism in NLP, revealing various biases inherent in NLP model development stages. This underscores the importance of considering racial fairness when applying model compression techniques, a key aspect of our study. Zhao et al. (2020) explored gender bias in multilingual embeddings, particularly in the context of cross-lingual transfer, a perspective relevant to our investigation into the fairness implications of compressing multilingual models.\n\nOrgad and Belinkov (2022) critiqued the evaluation methods for gender bias in NLP, emphasizing the need for extrinsic bias metrics, which resonates with our approach of using both intrinsic and extrinsic metrics to assess fairness in compressed language models. Blodgett et al. (2020) surveyed papers analyzing bias in NLP systems, finding inconsistencies and vagueness in motivations and techniques, which further justifies our comprehensive evaluation approach in examining the fairness of compressed models. Silva et al. (2021) analyzed gender and racial bias across several pre-trained transformers, using metrics that we also employ in our study to evaluate bias in compressed models.\n\nTal et al. (2022) investigated the connection between model size and gender bias, a study pertinent to our evaluation of different compression techniques on fairness metrics. Their findings on the varying impacts of model size on bias underscore the complexity of the relationship between model compression and fairness. Ahn et al. (2022) highlighted how knowledge distillation can amplify gender bias, offering insights into one of the compression techniques we scrutinize. Xu and Hu (2022) examined the effect of model compression on fairness, specifically focusing on toxicity and bias, which complements our broader analysis of fairness in language models post-compression. Finally, Goldfarb-Tarrant et al. (2021) and Mohammadshahi et al. (2022) provided critical insights into the correlation (or lack thereof) between intrinsic and extrinsic bias metrics and the forgotten aspects of compressed multilingual models, respectively, both of which are integral to the understanding and context of our work."

    mae_para, mc_rate, ari_greedy, ari_mat, ari_dot, f1 = cluster_eval([gt], [ref], [generated])
    print(1)
    # gt_df = text2label(gt)
    # pred_df = text2label(generated)
