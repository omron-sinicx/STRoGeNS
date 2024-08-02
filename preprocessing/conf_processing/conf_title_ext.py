import argparse
from pathlib import Path
import re
import requests
import json
import random
import pickle

import numpy as np
from tqdm import tqdm
import pandas as pd
from vllm import LLM, SamplingParams
from loguru import logger
import os


def extract_title(content):
    sampling_params = SamplingParams(temperature=0.9, max_tokens=50)

    prompt = f"""This text is the first 200 characters of the conference paper. Extract a paper title from the text.\nOutput is paper title only. Add \" ||| \" at the end of the extracted title.\nExamples:\n    
    {{# Zero-shot Natural Language Video Localization\n\nJinwoo Nam\\({{}}^{{1,*}}\\)\n\nDaechul Ahn\\({{}}^{{1,*}}\\)\n\nDongyeop Kang\\({{}}^{{2,\\lx@sectionsign}}\\)\n\nSeong Jong Ha\\({{}}^{{3}}\\)\n\nJonghyun Choi\\({{}}^{{1,\\dagger}}\\)\n\n\\({{}} }} -> Title: Zero-shot Natural Language Video Localization |||
    {{# A Second look at Exponential and Cosine Step Sizes:\n\nSimplicity, Adaptivity, and Performance\n\nXiaoyu Li\n\nEqual contribution \\({{}}^{{1}}\\)Division of System Engineering, Boston University, Boston, MA, U'}} -> Title: A Second look at Exponential and Cosine Step Sizes: Simplicity, Adaptivity, and Performance ||| 
    \nExtract Paper title from the following one text.\nText:\n   {{{content} }} -> Title:"""
    preds = llm.generate(prompt, sampling_params)

    return preds[0].outputs[0].text.split("|||")[0]


def extract_each_section(content):
    sections = re.split(r"(?<!#)##(?!#)", content)

    paper_sections = {
        "Introduction": [],
        "Experiment": [],
        "Related Work": [],
        "Conclusion": [],
        "Acknowledgements": [],
        "Reference": [],
        "Abstract": [],
    }

    for section in sections:
        for tag in paper_sections.keys():
            if tag.lower() in section[:20].lower():
                section = re.sub(r"\\begin\{table\}.*?\\end\{table\}", "", section, flags=re.DOTALL)
                paper_sections[tag].append(section)
                if tag == "Related Work":
                    if "[MISSING_PAGE_FAIL:3]" in section:
                        return 1
    if (len(paper_sections["Related Work"]) == 0) or (len(paper_sections["Related Work"]) > 1):
        num_sections = len(paper_sections["Related Work"])
        print(f"The number of related work section is {num_sections}")

    # extract abstract
    if "abstract" in sections[0].lower():
        try:
            abst = sections[0].split("Abstract")[1]
            paper_sections["Abstract"] = abst
        except:
            pass
    else:
        pass
    return paper_sections


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


def find_author_citation(paragraph):
    author_list_multi_year = []
    # find Author year, year style name. to avoid two author as one author replace Author and Author to @readed token
    author_list_multi_year.extend(
        re.findall(
            r"[A-Z][A-Za-z\-]+ and,? [A-Za-z\-]+,? ?(?:et al\.,? )?(?:\(\d{4}[a-z]?, \d{4}[a-z]?\)|\d{4}[a-z]?, \d{4}[a-z]?)+",
            paragraph,
        )
    )
    readed_paragraph = re.sub(
        r"[A-Z][A-Za-z\-]+ and,? [A-Za-z\-]+,? ?(?:et al\.,? )?(?:\(\d{4}[a-z]?, \d{4}[a-z]?\)|\d{4}[a-z]?, \d{4}[a-z]?)+",
        "@readed",
        paragraph,
    )
    author_list_multi_year.extend(
        re.findall(
            r"(?<! and\s)[A-Z][A-Za-z\-]+,? (?:et al\.,? )?(?:\(\d{4}[a-z]?,? \d{4}[a-z]?\)|\d{4}[a-z]?, \d{4}[a-z]?)+",
            readed_paragraph,
        )
    )
    readed_paragraph = re.sub(
        r"[A-Z][A-Za-z\-]+,? (?:et al\.,? )?(?:\(\d{4}[a-z]?,? \d{4}[a-z]?\)|\d{4}[a-z]?, \d{4}[a-z]?)+",
        "@readed",
        readed_paragraph,
    )

    author_list = []
    # find Author and Author name. to avoid two author as one author replace Author and Author to @readed token
    author_list.extend(
        re.findall(
            r"[A-Z][A-Za-z\-]+ and,? [A-Za-z\-]+,? ?(?:et al\.,? )?(?:\(\d{4}[a-z]?\)|\d{4}[a-z]?)+(?!,\s\d{4})",
            readed_paragraph,
        )
    )
    readed_paragraph = re.sub(
        r"[A-Z][A-Za-z\-]+ and,? [A-Za-z\-]+,? ?(?:et al\.,? )?(?:\(\d{4}[a-z]?\)|\d{4}[a-z]?)+(?!,\s\d{4})",
        "@readed",
        readed_paragraph,
    )
    author_list.extend(
        re.findall(r"[A-Z][A-Za-z\-]+,? (?:et al\.,? )?(?:\(\d{4}[a-z]?\)|\d{4}[a-z]?)+", readed_paragraph)
    )

    # Author (and)? (et al.) year
    norm_author_set = set()
    for author in author_list_multi_year:
        norm_author = author.replace(",", "").replace("(", "").replace(")", "").split(" ")
        year1, year2 = norm_author[-2:]
        author_info = " ".join(norm_author[:-2])
        norm_author_set.add(f"{author_info} {year1}")
        norm_author_set.add(f"{author_info} {year2}")

        paragraph = paragraph.replace(author, f"[{author_info} {year1}]; [{author_info} {year2}]")

    for author in author_list:
        norm_author = author.replace(",", "").replace("(", "").replace(")", "")
        norm_author_set.add(norm_author)

        paragraph = paragraph.replace(author, f"[{norm_author}]")

    return paragraph, list(norm_author_set)


def separate2paragraph(related_work, err_log=0):
    paragraph_list = []
    if len(related_work) > 0:
        for paragraph in related_work[0].split("\n"):
            if (
                (len(paragraph) < 100)
                or ("Table" in paragraph[:10])
                or ("Figure" in paragraph[:10])
                or ("Footnote" in paragraph[:10])
            ):
                continue
            if paragraph[0].islower():
                try:
                    paragraph_list[-1] += paragraph
                except IndexError:
                    paragraph_list.append(paragraph)
                    err_log = 1
            else:
                paragraph_list.append(paragraph)
    return paragraph_list, err_log


def cite_extraction(rw_paragraph_list, conf_name):
    # return: {"paragraph":[], "citations": []}
    paragraph_list = []
    citation_list = []

    for paragraph in rw_paragraph_list:
        if conf_name in ["cvpr", "eccv", "iccv"]:
            paragraph, citations = find_digit_citation(paragraph)
        else:  # acl, emnlp, iclr, icml, naacl
            paragraph, citations = find_author_citation(paragraph)
            _, digit_citations = find_digit_citation(paragraph)

            if len(digit_citations) > 0:
                citations.extend(digit_citations)

        # if (len(citations) == 0):
        # continue

        paragraph_list.append(paragraph)
        citation_list.append(citations)
    return paragraph_list, citation_list


def title_extraction_from_ref(title_list, llm):
    sampling_params = SamplingParams(temperature=0.9, max_tokens=50)

    prompts = []
    for ref_txt in title_list:
        prompt = f"""Extract a paper title from the sentence.\nOutput is paper title only. Add \" ||| \" at the end of the extracted title.\nExamples:\n    * [36] Prannay Khosla, Piotr Teterwak, Chen Wang, Aaron Sarna, Yonglong Tian, Phillip Isola, 
                Aaron Maschinot, Ce Liu, and Dilip Krishnan. Supervised contrastive learning. In _NeurIPS_, 2020. -> Title:Supervised contrastive learning. ||| \n    * [37] Jonathan Krause, Michael Stark, Jia Deng, and Li Fei-Fei. 3d object 
                representations for fine-grained categorization. In _4th International IEEE Workshop on 3D Representation and Recognition (3dRR-13)_, Sydney, Australia, 2013. -> Title: 3d object representations for fine-grained 
                categorization. ||| \nExtract Paper title from the following one sentence.\n    {ref_txt} -> Title:"""
        prompts.append(prompt)

    preds = llm.generate(prompts, sampling_params)

    generated_list = []
    for pred in preds:
        response = pred.outputs[0].text
        response = response.replace("\n", "")
        response = response.split("|||")[0]
        generated_list.append(response)
    return generated_list


def generate_reference_dict(cites, identifier, ref_lists, llm):
    # extract title only matched reference and return dict which has identifier and title

    # remove overlap
    whole_cites = set()
    for cite in cites:
        whole_cites = whole_cites.union(cite)

    # find matched id and unmatched
    matched_ref_list = []
    matched_id_list = []
    unmatched_id_list = []
    for cite in whole_cites:
        try:
            ref_id = identifier.index(cite)
            matched_ref_list.append(ref_lists[ref_id])
            matched_id_list.append(cite)
        except:
            unmatched_id_list.append(cite)

    estimeted_title_list = title_extraction_from_ref(matched_ref_list, llm)

    ref_dict = {}
    for m_id, title in zip(matched_id_list, estimeted_title_list):
        ref_dict[m_id] = title
    for m_id in unmatched_id_list:
        ref_dict[m_id] = "Could not matching identifier"
    return ref_dict


def identifier_extraction_from_ref(ref_lists):
    identifiers = []
    ref_list_new = []
    for ref_txt in ref_lists:
        digit_match = re.match(r"\* \[\d+\]", ref_txt)
        author_match = re.match(
            r"\* [?[A-Za-z\-]+ (?:(et al.,? ?)|(?:and [A-Za-z\-]+ ?))??\(?\d{4}[a-z]?\)?]?", ref_txt
        )
        if digit_match is not None:
            identifier = digit_match.group().split(" ")[-1]

            # remove []
            identifier = identifier[1:-1]
        elif author_match is not None:
            identifier = author_match.group()[2:]
            identifier = (
                identifier.replace(",", "")
                .replace("(", "")
                .replace(")", "")
                .replace("[", "")
                .replace("]", "")
                .replace("et al.2", "et al. 2")
            )
        else:
            print("Could not find identifier")
            continue
        ref_list_new.append(ref_txt)
        identifiers.append(identifier)
    return identifiers, ref_list_new


def start_from_author(line):
    return (re.match(r"\* \[?[A-Za-z\-]+", line) is not None) or (re.match(r"\* \[\d+\]", line) is not None)


def extract_ref_paper_title(references, related_work, llm):
    # extract identifier and titles from references
    ref_lists = list(filter(start_from_author, references[0].split("\n")))

    identifier, ref_lists = identifier_extraction_from_ref(ref_lists)

    # extract_title
    ref_dict = generate_reference_dict(related_work[1], identifier, ref_lists, llm)

    title_dict = {}
    paper_list = []
    for text, citations in zip(related_work[0], related_work[1]):
        paper_info_for_cite = {}
        # title
        for cite in citations:
            paper_info_for_cite[cite] = ref_dict[cite]
            title_dict[cite] = ref_dict[cite]
        paper_list.append([text, citations, paper_info_for_cite])

    return paper_list, title_dict


def evaluation(paper_info):
    mat_err = 0  # matching error. Fail to extract matched paper.
    sucess = 0
    for papers in paper_info:
        if papers[1] != []:
            for key in papers[2]:
                if papers[2][key] == "Could not matching identifier":
                    mat_err += 1
                else:
                    sucess += 1
        else:
            print("Any references")
    return mat_err, sucess


def processing_papers(paper_list, llm, conf_name):
    dataset = []
    sucess_rate_list = []
    parse_error_titles = []
    total_processed_paper = 0
    total_suqu = 0
    total_mat_err = 0
    for iter, paper_path in enumerate(paper_list):
        # read content from mark down file
        with open(paper_path, "r") as file:
            # read file content
            paper_content = file.read()

        # title extraction. It
        title = extract_title(paper_content[:200])
        print(title)

        # separate section with ##
        sections = extract_each_section(paper_content)
        if sections == 1:
            parse_error_titles.append(title)
            continue

        rw_paragraph_list, err_log = separate2paragraph(sections["Related Work"])

        if err_log == 1:
            logger.info(f"miss {title} {iter}/{len(paper_list)}")
            continue

        if (len(rw_paragraph_list) > 1) and (len(sections["Reference"]) > 0) and (len(sections["Abstract"]) > 0):
            # obtain paragraph and citation markers such as [\d+] or Author et al. year
            related_work = cite_extraction(rw_paragraph_list, conf_name)

            # obtain paper title based on reference and extract info. by semantic scholar
            paper_info, ref_title_dict = extract_ref_paper_title(sections["Reference"], related_work, llm)

            # success rate calculation
            mat_err, sucess = evaluation(paper_info)

            if sucess == 0:
                continue
            total_suqu += sucess
            total_mat_err += mat_err
            sucess_rate = sucess / (sucess + mat_err)
            sucess_rate_list.append(sucess_rate)
            print(f"sucess rate {sucess_rate}")

            total_processed_paper += 1
            dataset.append(
                {"title": title, "abst": sections["Abstract"], "related_work": paper_info, "ref_title": ref_title_dict}
            )
        else:
            pass
        if (iter % 1000) == 0:
            logger.info(f"{iter}/{len(paper_list)}")
    print(f"Total processed {total_processed_paper}")
    return dataset, sucess_rate_list, total_processed_paper, total_suqu, total_mat_err


def generate_paper_list(conf, conf_list):
    paper_list = []
    total_papers = 0
    for conf_dir in conf_list:
        if conf_dir.is_dir():
            if conf in conf_dir.stem:
                if (conf == "acl") and "naacl" in conf_dir.stem:
                    continue
                paper_list.extend(sorted(conf_dir.glob("*.mmd")))
                total_papers += len(sorted(conf_dir.glob("*.mmd")))
    return paper_list, total_papers


def parse_args():
    parser = argparse.ArgumentParser(description="Train data path")
    parser.add_argument("--data_dir", default="./data/STRoGeNS-conf22/mds ")
    parser.add_argument("--output_dir", default="./data/STRoGeNS-arXiv22/rw")
    parser.add_argument("--log_dir", default="./logs/STRoGeNS-conf22/rw")
    parser.add_argument("--debug", default=False, action="store_true")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    ## make output path
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    Path(args.log_dir).parent.mkdir(parents=True, exist_ok=True)
    logger.add(f"{args.log_dir}.log", rotation="500 MB")

    ## Prepare llama
    llm = LLM(model="meta-llama/Llama-2-7b-chat-hf", max_num_batched_tokens=4096)
    # llm = LLM(model="HuggingFaceH4/zephyr-7b-beta")

    conf_list = sorted(Path(args.data_dir).iterdir())

    dataset_list = []
    suqucessed_whole = []
    for conf in conf_list:
        logger.info(f"{conf} start processing")

        # Collect conf. papers
        paper_list, total_papers = generate_paper_list(conf, conf_list)

        if args.debug:
            random.shuffle(paper_list)
            paper_list = paper_list[:10]
            total_papers = 10

        print(f"{conf} {total_papers}")
        logger.info(f"{conf} total {total_papers}")

        # Processing conf paers
        dataset, suqucessed, total_processed_paper, total_succ, total_mat_err = processing_papers(paper_list, llm, conf)

        logger.info(f"{conf} total processed {total_processed_paper}")
        logger.info(f"{conf} matching error {total_mat_err}, suqu {total_succ}")

        # Save conf. papers
        with output_path.joinpath(f"dataset_{conf}.pkl").open("wb") as f:
            pickle.dump(dataset, f)
        with output_path.joinpath(f"suqucessed_{conf}.pkl").open("wb") as f:
            pickle.dump(suqucessed, f)

        suqucessed_whole.extend(suqucessed)
        dataset_list.extend(dataset)

        avg_suqucessed = np.array(suqucessed).mean()
        logger.info(f"{conf} avg suqucessed {avg_suqucessed}")

    # Save whole results
    with output_path.joinpath(f"dataset.pkl").open("wb") as f:
        pickle.dump(dataset_list, f)
    with output_path.joinpath(f"suqucessed.pkl").open("wb") as f:
        pickle.dump(suqucessed_whole, f)
