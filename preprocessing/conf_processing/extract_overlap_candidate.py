from thefuzz import process

import pandas as pd
from datasets import load_from_disk


if __name__=="__main__":
    conf23 = load_from_disk("hg_dataset/conf23_rw")

    conf_23_title_list = []
    for data in conf23['input']:
        conf_23_title_list.append(data.split("\n")[1][7:].strip())

    arxiv22 = load_from_disk("hg_dataset/arxiv_rw")
    arxiv22_title_list = []
    for data in arxiv22['input']:
        arxiv22_title_list.append(data.split("Abstract:")[0].split("Title:")[1].replace("\n", ""))

    conf_title_cand = []
    conf_idx = []
    arxiv_title_cand = []
    sim_score = []
    for idx, conf_sample in enumerate(conf_23_title_list):
        sim_sample = process.extractOne(conf_sample, arxiv22_title_list)

        if (sim_sample[1] >= 88) and (sim_sample[1] < 90):
            print(conf_sample)
            print(sim_sample)
            pass

        if sim_sample[1] >= 90:
        
            conf_title_cand.append(conf_sample)
            conf_idx.append(idx)
            arxiv_title_cand.append(sim_sample[0])
            sim_score.append(sim_sample[1])
            

    cand_df = pd.DataFrame({"Conf": conf_title_cand, "arxiv": arxiv_title_cand, "similarity": sim_score, "conf_idx": conf_idx})
    cand_df.to_csv('overlap_candidates.csv')
    print(1)

# "CLIP-Sculptor: Zero-Shot Generation of High-Fidelity and Diverse Shapes from Natural Language", "Detecting Objects with Context-Likelihood Graphs and Graph Refinement"