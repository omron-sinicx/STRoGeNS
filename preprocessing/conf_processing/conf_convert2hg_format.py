import argparse
from pathlib import Path
import pickle
import re

from datasets import Dataset as hg_Dataset


def convertdict(conf_datadir):
    dataset_dict = {"input": [], "output": [], "references": [], "ref_numbers": []}
    dataset_paths = sorted(Path(conf_datadir).iterdir())

    total_dataset_size = 0
    for dataset_path in dataset_paths:
        with open(dataset_path, "rb") as f:
            dataset = pickle.load(f)
        total_dataset_size += len(dataset)
        for data in dataset:
            input_text = ""
            ref_text = []
            output_text = " "
            title = data["title"]
            try:
                abst = data["abst"].replace("\n", "")
            except:
                continue
            input_text = f"Target paper:\nTitle: {title}\nAbstract: {abst}\n\n"

            if len(data["related_work"]) < 2:
                continue
            if len(data["related_work"]) > 8:
                continue
            for ref in data["related_work"]:
                text = ref[0]
                output_text += f"{text}\n\n"

            output_org = output_text
            ref_list = []
            num_ref = 1
            for ref_key, ref in data["ref_title"].items():
                if (type(ref) is not str) and (ref["abstract"] is not None):
                    ref_abst = ref['abstract'].replace("\n", "")
                    ref_text.append(f"[{num_ref}] {ref['title']}, Abstract: {ref_abst}")
                    # ref_list.append(ref_key)
                    output_text = output_text.replace(f"[{ref_key}]", f"<<{num_ref}>>")

                    ref_list.append(num_ref)

                    num_ref += 1
                else:
                    #     input_text += f"[{ref_key}] None\n\n"
                    output_text = output_text.replace(f"[{ref_key}]", f"")
            if num_ref > 2:
                output_text = output_text.replace("<<", "[").replace(">>", "]")
                dataset_dict["input"].append(input_text)
                dataset_dict["output"].append(output_text)
                dataset_dict["references"].append(ref_text)
                dataset_dict["ref_numbers"].append(ref_list)
    return dataset_dict


def parse_args():
    parser = argparse.ArgumentParser(description="Train data path")
    parser.add_argument("--data_dir", default="data/STRoGeNS-conf22/rw_wabst")
    parser.add_argument("--output_dir", default="data/STRoGeNS-conf22/hg_format")
    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = parse_args()
    
    save_path = Path(args.output_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    dataset_dict = convertdict(args.data_dir)
    ds = hg_Dataset.from_dict(dataset_dict)
    print(ds)
    ds.save_to_disk(save_path)
