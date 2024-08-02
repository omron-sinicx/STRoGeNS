import argparse
from pathlib import Path
import json
import requests
import pickle

import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams
from loguru import logger


def extract_title(content, llm):
    sampling_params = SamplingParams(temperature=0.9, max_tokens=50)

    prompt = f"""This text is the first 200 characters of the conference paper. Extract a paper title from the text.\nOutput is paper title only. Add \" ||| \" at the end of the extracted title.\nExamples:\n    
    Jahangiri, M., Sacharidis, D., and Shahabi, C. 2005. SHIFT-SPLIT: I/O efficient maintenance of wavelet-transformed multidimensional data. In SIGMOD '05. 275–286. -> Title: SHIFT-SPLIT: I/O efficient maintenance of wavelet-transformed multidimensional data. |||
    Poon, C. 2003. Dynamic orthogonal range queries in OLAP. Theoretical Computer Science 296, 3, 487–510. -> Dynamic orthogonal range queries in OLAP. ||| 
    \nExtract Paper title from the following one text.\nText:\n   {{{content} }} -> Title:"""
    preds = llm.generate(prompt, sampling_params)

    return preds[0].outputs[0].text.split("|||")[0]


def search_title(cite_list, bib_info, llm):
    title_list = {}
    for cite in cite_list:
        ref_id = cite["ref_id"]
        ref_info = bib_info[ref_id]
        try:
            open_alex_id = ref_info["ids"]["open_alex_id"].split("/")[-1]
            r = requests.get(f"https://api.openalex.org/works/{open_alex_id}").json()
            title = r["title"]
        except:
            title = extract_title(ref_info["bib_entry_raw"], llm)
        title_list[ref_id] = title
    return title_list


def parse_args():
    parser = argparse.ArgumentParser(description="Train data path")
    parser.add_argument("--data_dir", default="./data/STRoGeNS-arXiv22/rawdata")
    parser.add_argument("--output_dir", default="./data/STRoGeNS-arXiv22/rw")
    parser.add_argument("--log_dir", default="./logs/STRoGeNS-arXiv22/rw")
    parser.add_argument("--n_dir", default=0, type=int)
    parser.add_argument("--debug", default=False, action="store_true")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    ## make output path
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    arxiv_dir_list = sorted(Path(args.data_dir).iterdir())
    
    Path(args.log_dir).parent.mkdir(parents=True, exist_ok=True)
    logger.add(f"{args.log_dir}.log", rotation="500 MB")

    chunk_size = 10000

    total_papers = 0

    ## Prepare llama
    llm = LLM(model="meta-llama/Llama-2-7b-chat-hf", max_num_batched_tokens=4096)
    # llm = None

    chunk_num = 0
    for arxiv_dir in tqdm(arxiv_dir_list[args.n_dir:]):
        data_paths = sorted(arxiv_dir.glob("*.jsonl"))

        iteration = 0

        logger.info(f"Total {len(data_paths)}")
        dataset = []
        for data_path in data_paths:
            # load data
            with data_path.open("r") as f:
                jsonl_data = [json.loads(l) for l in f.readlines()]

            total_papers += len(jsonl_data)

            for paper_info in jsonl_data:
                # paper info
                title = paper_info["metadata"]["title"]
                abst = paper_info["metadata"]["abstract"]
                arxiv_id = paper_info["metadata"]["id"]
                license_info = paper_info["metadata"]["license"]
                categories = paper_info["metadata"]["categories"]
                main_category = paper_info["discipline"]

                bib_info = paper_info["bib_entries"]

                # extract related work
                related_work_list = []
                citation_count = 0
                for para in paper_info["body_text"]:
                    if para["section"] is not None:
                        if ("related work" in para["section"].lower()) or ("previous work" in para["section"].lower()):
                            citation_count += len(para["cite_spans"])
                            paper_info_dict = search_title(para["cite_spans"], bib_info, llm)
                            related_work_list.append([para["text"], para["cite_spans"], paper_info_dict])

                if citation_count > 0:
                    sample = {
                        "title": title,
                        "abst": abst,
                        "related_work": related_work_list,
                        "arxiv_info": {"id": arxiv_id, "categories": [main_category, categories], "license": license_info},
                    }
                    dataset.append(sample)
                    iteration += 1
                    if (iteration % chunk_size) == 0:
                        with open(f"{output_path}/{arxiv_dir.stem}_{chunk_num}.pkl", "wb") as f:
                            pickle.dump(dataset, f)
                        dataset = []
                        chunk_num += 1

                    if (iteration % 1000) == 0:
                        logger.info(f"{iteration}")
        logger.info(f"{arxiv_dir.stem} total {len(dataset)} papers")
        if len(dataset) > 0:
            with open(f"{output_path}/{arxiv_dir.stem}_{chunk_num}.pkl", "wb") as f:
                pickle.dump(dataset, f)
