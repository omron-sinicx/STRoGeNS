import argparse
import time
import requests
import random
import os
import sys

from selenium import webdriver
from slugify import slugify
import jsonlines

from urls import conference_url
from retrieve_functions import retrieve

# def download_one(pdfname, pdfurl):
#     if pdfurl != None:
#         pdfname_slug = slugify(pdfname)
#         if os.path.isfile(args.save_dir + "/" + pdfname_slug + ".pdf"):
#             print("existed", i, "\t", pdfname, "\t", pdfurl)
#         else:
#             print(i, "\t", pdfname, "\t", pdfurl)
#             data = requests.get(pdfurl, timeout=80, headers=headers).content
#             with open(args.save_dir + "/" + pdfname_slug + ".pdf", "wb") as f:
#                 f.write(data)
#             _ = time.sleep(random.uniform(4, 5))  # for anti-anti-crawler purpose

"""
some variables needed to be set up by users

conference urls examples:
cvpr: https://openaccess.thecvf.com/CVPR2020 (CVPR 2020)
eccv: https://openaccess.thecvf.com/ECCV2018 (ECCV 2018)
eccv: https://www.ecva.net/papers.php (ECCV 2020) (changed in 2020)
iccv: https://openaccess.thecvf.com/ICCV2019 (ICCV 2019)
icml: http://proceedings.mlr.press/v119/ (ICML 2020)
neurips: https://papers.nips.cc/paper/2020 (NeurIPS 2020)
iclr: https://openreview.net/group?id=ICLR.cc/2021/Conference (ICLR 2021)
siggraph: https://dl.acm.org/toc/tog/2020/39/4 (SIGGRAPH 2021)

"""

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Train data path")
    parser.add_argument("--conference", default="acl2019", type=str)
    parser.add_argument("--save_dir", default="data", type=str)
    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = parse_args()

    # import pdb
    # pdb.set_trace()

    conference = args.conference
    year = conference[-4:]
    url = conference_url[conference]  # the conference url to download papers from
    
    if int(year) > 2022:
        save_dir = f"{args.save_dir}/STRoGeNS-conf23/pdfs"
    else:
        save_dir = f"{args.save_dir}/STRoGeNS-conf22/pdfs"

    os.makedirs(save_dir, exist_ok=True)

    server = 'http://selenium-chrome:4444/wd/hub'
    options = webdriver.ChromeOptions()
    driver = webdriver.Remote(command_executor=server, options=options)

    if isinstance(url, list):
        pdfurllist = []
        pdfnamelist = []
        abstlist = []
        for ele in url:
            driver.get(ele)
            print("Retrieving pdf urls. This could take some time...")
            pdfurl, pdfname, abst = retrieve(driver, conference)
            if len(pdfurl) != len(pdfname):
                print(1)

            pdfurllist = pdfurllist + pdfurl
            pdfnamelist = pdfnamelist + pdfname
            abstlist = abstlist + abst

    else:
        driver.get(url)
        pdfurllist, pdfnamelist, abstlist = retrieve(driver, conference)
    print("scraped: ", len(pdfurllist))
    import pdb
    pdb.set_trace()
    # check the retrieved urls
    print("The first 5 pdf urls:\n")
    for i in range(5):
        print(pdfurllist[i])
    print("\nThe last 5 pdf urls:\n")
    for i in range(5):
        print(pdfurllist[-(i + 1)])
    print("=======================================================")

    # check the retrieved paper titles
    print("The first 5 pdf titles:\n")
    for i in range(5):
        print(pdfnamelist[i])
    print("\nThe last 5 pdf titles:\n")
    for i in range(5):
        print(pdfnamelist[-(i + 1)])

    pdf_alls = [{"title": title, "url": url, "abst": abst} for title, url, abst in zip(pdfnamelist, pdfurllist, abstlist)]

    with jsonlines.open(conference + ".jsonl", "w") as writer:
        writer.write_all(pdf_alls)

    print("The number of papers is ", len(pdfnamelist))
    assert len(pdfnamelist) == len(
        pdfurllist
    ), "The number of titles and the number of urls are not matched. \
                                                You might solve the problem by checking the HTML code in the \
                                                website yourself or you could ask the author by raising an issue."

    # download the papers one by one. The files are named after the titles (guaranteed to be valid file name after processed by slugify.)
    print("Start downloading")
    headers = {"user-agent": "Mozilla/5.0 (Windows NT 10.0; WOW64; rv:57.0) Gecko/20100101 Firefox/57.0"}

    download_pdfs = True
    if download_pdfs:
        for i, url in enumerate(pdfurllist):
            if url != None:
                pdfname = slugify(pdfnamelist[i])
                if os.path.isfile(args.save_dir + "/" + pdfname + ".pdf"):
                    print("existed", i, "\t", pdfnamelist[i], "\t", pdfurllist[i])
                else:
                    print(i, "\t", pdfnamelist[i], "\t", pdfurllist[i])
                    data = requests.get(pdfurllist[i], timeout=80, headers=headers).content
                    with open(args.save_dir + "/" + pdfname + ".pdf", "wb") as f:
                        f.write(data)
                    _ = time.sleep(random.uniform(4, 5))  # for anti-anti-crawler purpose

    input_args = [[name, url] for name, url in zip(pdfnamelist, pdfurllist)]

    download_pdfs = False
    if download_pdfs:
        for i, url in enumerate(pdfurllist):
            if url != None:
                pdfname = slugify(pdfnamelist[i])
                if os.path.isfile(args.save_dir + "/" + pdfname + ".pdf"):
                    print("existed", i, "\t", pdfnamelist[i], "\t", pdfurllist[i])
                else:
                    print(i, "\t", pdfnamelist[i], "\t", pdfurllist[i])
                    data = requests.get(pdfurllist[i], timeout=80, headers=headers).content
                    with open(args.save_dir + "/" + pdfname + ".pdf", "wb") as f:
                        f.write(data)
                    _ = time.sleep(random.uniform(4, 5))  # for anti-anti-crawler purpose
