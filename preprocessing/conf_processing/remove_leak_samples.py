import pandas as pd
from datasets import load_from_disk
from datasets import Dataset

if __name__=="__main__":
    overlap_papers = pd.read_csv("preprocessing/annotations/remove_overlap.csv")

    conf23 = load_from_disk("hg_dataset/conf23_rw")
    conf23_df = conf23.to_pandas()

    remove_papers_idx = overlap_papers[overlap_papers['FALSE'] == False]['conf_idx'].values

    cleaned_conf23 = conf23_df[~conf23_df.index.isin(remove_papers_idx)]

    
    conf23_v2 = Dataset.from_pandas(cleaned_conf23)
    conf23_v2.save_to_disk('hg_dataset/conf23_rw_v2')

    # selected_samples = pd.read_csv("novelty_annotation/datasets/100_sampled_df.csv")

    # additional_annot = cleaned_conf23[~cleaned_conf23.index.isin(selected_samples['Unnamed: 0.1'].values)].sample(6)
    # additional_annot = additional_annot[['output']].rename(columns={"output":"gt"})
    # additional_annot['label'] = None
    # additional_annot.to_json('novelty_annotation/datasets/additional.jsonl', force_ascii=False, lines=True, orient='records')

    print(1)
