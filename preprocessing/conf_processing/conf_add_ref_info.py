from pathlib import Path
import argparse
import pickle
import requests
import json
import time
import difflib

from requests.exceptions import HTTPError

from tqdm import tqdm
from loguru import logger

S2_API_KEY = "{S2APIKEY}"


def send_request(end_point, params, headers, max_retries=5):
    retry_wait = 1  # initial weiting time（second）
    for i in range(max_retries):
        try:
            response = requests.get(url=end_point, params=params, headers=headers)
            # raise HTTPError
            response.raise_for_status()  # HTTPError
            return response  
        except HTTPError as e:
            if e.response.status_code == 429:
                print(f"Too mutch request wait {retry_wait} seconds...")
                time.sleep(retry_wait)
                retry_wait *= 2  # 待機時間を指数的に増加
            else:
                break  # 429以外のHTTPErrorは再送信しない

    return 1


def search_paper_info(title, squcess, search_err, no_abst):
    # The accecible values is described in following website. Please look fields in the site.
    # https://api.semanticscholar.org/api-docs/#tag/Paper-Data/operation/post_graph_get_papers
    endpoint = "https://api.semanticscholar.org/graph/v1/paper/search"
    fields = ("title", "openAccessPdf", "abstract")
    headers = {
        "X-API-KEY": S2_API_KEY,
    }
    params = {"query": title, "fields": ",".join(fields), "limit": 1}

    response = send_request(endpoint, params, headers)
    if response == 1:
        paper_info = "Could not find matched title"
        search_err += 1
    else:
        r_dict = json.loads(response.text)

        try:
            if r_dict["total"] == 0:
                paper_info = "Could not find matched title"
                search_err += 1
            else:
                paper_info = r_dict["data"][0]
                simirality = difflib.SequenceMatcher(None, title, paper_info["title"]).ratio()
                if simirality < 0.5:
                    paper_info = "Could not find matched title"
                    search_err += 1
                elif paper_info["abstract"] is None:
                    no_abst += 1
                else:
                    squcess += 1
        except KeyError:
            paper_info = "Could not find matched paper"
            search_err += 1
    return paper_info, squcess, search_err, no_abst


def parse_args():
    parser = argparse.ArgumentParser(description="Train data path")
    parser.add_argument("--data_dir", default="data/STRoGeNS-conf22/rw")
    parser.add_argument("--output_dir", default="data/STRoGeNS-conf22/rw_wabst")
    parser.add_argument("--log_dir", default="logs/STRoGeNS-conf22/rw_wabst")
    parser.add_argument("--debug", default=False, action="store_true")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # make output path
    output_path = Path(f"{args.output_dir}/{args.run_name}")
    output_path.mkdir(parents=True, exist_ok=True)

    Path(args.log_dir).parent.mkdir(parents=True, exist_ok=True)
    logger.add(f"{args.log_dir}.log", rotation="500 MB")

    conf_dirs = sorted(Path(args.data_dir).glob(f"dataset_*"))
    for conf_dir in conf_dirs:
        match_err, search_err, no_abst, squcess = 0, 0, 0, 0

        # load data
        with conf_dir.open("rb") as f:
            dataset = pickle.load(f)

        logger.info(f"{conf_dir.stem} start processing")
        logger.info(f"Total num of sample is {len(dataset)}")

        # [title, abst, related_work, citations, citaion_info]
        for data in tqdm(dataset, position=1):
            for cite, title in tqdm(data["ref_title"].items(), position=0):
                if title != "Could not matching identifier":
                    try:
                        paper_info, squcess, search_err, no_abst = search_paper_info(
                            title, squcess, search_err, no_abst
                        )
                    except:
                        time.sleep(10)
                        paper_info, squcess, search_err, no_abst = search_paper_info(
                            title, squcess, search_err, no_abst
                        )
                    data["ref_title"][cite] = paper_info
                else:
                    match_err += 1
            data["squcess_rate"] = squcess / (squcess + search_err + no_abst + match_err)

        squcess_rate = squcess / (squcess + search_err + no_abst + match_err)
        logger.info(f"squcess, {squcess}, search_err, {search_err}, no_abst, {no_abst}, match, {match_err}")
        logger.info(f"squcess_rate {squcess_rate}")

        conf = conf_dir.stem.split("_")[-1]
        # Save conf. papers
        with output_path.joinpath(f"dataset_{conf}.pkl").open("wb") as f:
            pickle.dump(dataset, f)

        logger.info(f"finish {conf}")
