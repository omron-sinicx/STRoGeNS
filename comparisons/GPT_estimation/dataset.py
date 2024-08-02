from pathlib import Path
import pickle
import math


def convertdict(conf_datadir, max_tokens):
    dataset_dict = {"input": [], "output": [], "references": []}
    dataset_paths = sorted(Path(conf_datadir).iterdir())[1:]
    for dataset_path in dataset_paths:
        with open(dataset_path, "rb") as f:
            dataset = pickle.load(f)
        for data in dataset:
            input_text = ""
            output_text = " "
            title = data["title"]
            try:
                abst = data["abst"].replace("\n", "")
            except:
                continue
            
            abst_max_word = round(256 / math.sqrt(2))
            abst = " ".join(abst.split()[:abst_max_word])

            input_text += f"Target paper:\nTitle: {title}\nAbstract: {abst}\n\nReferences:\n"

            if len(data["related_work"]) < 2:
                continue
            if len(data["related_work"]) > 8:
                continue
            for ref in data["related_work"]:
                text = ref[0]
                output_text += f"{text}\n\n"

            ref_list = []
            num_ref = 1
            one_ref_max_word = round((max_tokens / len(data['ref_title'].keys())) / math.sqrt(2))
            for ref_key, ref in data["ref_title"].items():
                if (type(ref) is not str) and (ref['abstract'] is not None):
                    ref_text = f"[{num_ref}] {ref['title']}, Abstract: {ref['abstract']}"
                    ref_text = ' '.join(ref_text.split()[:one_ref_max_word]) + "\n\n"
                    input_text += ref_text
                    output_text = output_text.replace(f"[{ref_key}]", f"<<{num_ref}>>")
                    ref_list.append(num_ref)

                    num_ref += 1
                else:  
                #     input_text += f"[{ref_key}] None\n\n"
                    output_text = output_text.replace(f"[{ref_key}]", f"")

            import tiktoken
            enc = tiktoken.encoding_for_model("gpt-3.5-turbo-1106")
            user_prompt = enc.encode(input_text)
            if len(user_prompt) > max_tokens + 256:
                print(1)

            output_text = output_text.replace("<<", "[").replace(">>", "]")
            dataset_dict["input"].append(input_text)
            dataset_dict["output"].append(output_text)
            dataset_dict["references"].append(ref_list)
    return dataset_dict


if __name__ == "__main__":
    from datasets import Dataset as hg_Dataset

    dataset_dict = convertdict("/workdir/output/conferece2023_relatedwork")
    ds = hg_Dataset.from_dict(dataset_dict)
    print(1)
