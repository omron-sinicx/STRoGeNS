import argparse
from pathlib import Path
import pickle
import re

from datasets import Dataset as hg_Dataset


def convertdict_unarxiv(conf_datadir):
    dataset_dict = {"input": [], "output": [], "references": [], "ref_numbers": []}
    dataset_paths = sorted(Path(conf_datadir).iterdir())
    import pdb 
    pdb.set_trace()
    for dataset_path in dataset_paths:
        with open(dataset_path, "rb") as f:
            dataset = pickle.load(f)

        for data in dataset:
            input_text = ""
            ref_text = []
            output_text = " "
            title = data["title"]

            if len(data["related_work"]) < 2:
                continue
            if len(data["related_work"]) > 8:
                continue

            try:
                abst = data["abst"].replace("\n", "")
            except:
                continue
            
            input_text = f"Target paper:\nTitle: {title}\nAbstract: {abst}\n\n"

            ref_dict = {}
            for ref in data["related_work"]:
                text = ref[0].replace("\n", "")
                for ref_id in ref[2]:
                    if ref_id not in ref_dict:
                        ref_dict[ref_id] = ref[2][ref_id]
                output_text += f"{text}\n\n"

            ref_list = []
            num_ref = 1
            for ref_key, ref in ref_dict.items():
                if (type(ref) is not str) and (ref["abstract"] is not None):
                    ref_abst = ref['abstract'].replace("\n", "")
                    ref_text.append(f"[{num_ref}] {ref['title']}, Abstract: {ref_abst}")
                    output_text = output_text.replace(f"{{{{cite:{ref_key}}}}}", f"<<{num_ref}>>")

                    output_text = re.sub(r'{{[a-z]*:[0-9a-zA-Z-]*}}', 'Form', output_text)

                    ref_list.append(num_ref)

                else:
                    output_text = output_text.replace(f"{{{{cite:{ref_key}}}}}", f"")
            
            if (num_ref > 2):
                output_text = output_text.replace("<<", "[").replace(">>", "]")
                dataset_dict["input"].append(input_text)
                dataset_dict["output"].append(output_text)
                dataset_dict["references"].append(ref_text)
                dataset_dict["ref_numbers"].append(ref_list)
    return dataset_dict


def parse_args():
    parser = argparse.ArgumentParser(description="Train data path")
    parser.add_argument("--data_dir", default="data/STRoGeNS-arXiv22/rw_wabst")
    parser.add_argument("--output_dir", default="data/STRoGeNS-arXiv22/hg_format")
    args = parser.parse_args()
    return args


if __name__=="__main__":
    args = parse_args()
        
    save_path = Path(args.output_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    dataset_dict = convertdict_unarxiv(args.data_dir)
    ds = hg_Dataset.from_dict(dataset_dict)

    print(ds)
    ds.save_to_disk(save_path)
