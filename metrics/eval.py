from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .structure_eval import paragraph_eval
from .rouge155 import rouge_reference_overall


def evalate_whole_metrics(gt_list, generated_list, debug=False):
    rouge1 = []
    rouge2 = []
    rougeL = []
    mae_para_list = []
    mc_rate_list = []
    f1_list = []
    ari_greedy_list = []
    ari_mat_list = []
    ari_dot_list = []
    for gt, generated in tqdm(zip(gt_list, generated_list), total=len(gt_list)):

        rouge = rouge_reference_overall([gt], [generated])
        rouge1.append(rouge["rouge-1-f"])
        rouge2.append(rouge["rouge-2-f"])
        rougeL.append(rouge["rouge-l-f"])

        mae_para, mc_rate, f1, ari_greedy, ari_mat, ari_dot = paragraph_eval(gt, generated)

        if debug:
            rouge1_score = rouge["rouge-1-f"]
            rouge2_score = rouge["rouge-2-f"]
            rougel_score = rouge["rouge-l-f"]
            print("rouge 1, rouge 2, rouge l, f1, ARIgreed, ari_mat, ari_dot, mae, micitation")
            print(
                f"{rouge1_score:.3f}& {rouge2_score:.3f}& {rougel_score:.3f}& {f1:.3f}& {ari_greedy:.3f}& {ari_mat:.3f} & {ari_dot:.3f}& {mae_para:.3f}& {mc_rate:.3f}"
            )
            print(1)
        mae_para_list.append(mae_para)
        mc_rate_list.append(mc_rate)
        f1_list.append(f1)
        ari_greedy_list.append(ari_greedy)
        ari_mat_list.append(ari_mat)
        ari_dot_list.append(ari_dot)
    return (
        np.array(rouge1),
        np.array(rouge2),
        np.array(rougeL),
        np.array(mae_para_list),
        np.array(mc_rate_list),
        np.array(f1_list),
        np.array(ari_greedy_list),
        np.array(ari_mat_list),
        np.array(ari_dot_list),
    )


if __name__ == "__main__":
    data_path = Path("/workdir/output/cond_pred/bart_conf_5.csv")

    df = pd.read_csv()

    generated_list = df["pred"].values
    ref_list = df["ref"].values
    gt_list = df["gt"].values

    rouge1, rouge2, rougeL, mae_para_list, mc_rate_list, f1_list, ari_greedy_list, ari_mat_list, ari_dot_list = (
        evalate_whole_metrics(gt_list[:10], generated_list[:10])
    )
    print("rouge 1, rouge 2, rouge l, mae, micitation, f1, ARIgreed, ari_mat, ari_dot")
    print(
        f"{rouge1.mean():.3f}& {rouge2.mean():.3f}& {rougeL.mean():.3f}& {f1_list.mean():.3f}& {ari_greedy_list.mean():.3f}& {ari_mat_list.mean():.3f} & {ari_dot_list.mean():.3f}& {mae_para_list.mean():.3f}& {mc_rate_list.mean():.3f}"
    )

    df = pd.DataFrame({"pred": generated_list, "gt": gt_list, "ref": ref_list, 'rouge1': rouge1, 'rouge2': rouge2, 'rougel': rougeL, "f1": f1_list, 'ari_greed': ari_greedy_list, 'ari': ari_mat_list, 'ari_dot': ari_dot_list, 'mae': mae_para_list, 'mc': mc_rate_list})
    df.to_csv(f"{data_path.parent}/{data_path.stem}_eval.csv")

    # rouge_scores = rouge_reference_overall(gt_list[:10], generated_list[:10])
