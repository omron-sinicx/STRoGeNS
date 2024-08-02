import pdb
import re

from tqdm import tqdm
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup


def retrieve(driver, conference):
    retrieval_functions = {
        "cvpr": retrieve_from_cvpr,
        "iccv": retrieve_from_cvpr,
        "eccv": retrieve_from_eccv,
        "icml": retrieve_from_icml,
        "acl": retrieve_from_acl,
        "emnlp": retrieve_from_emnlp,
        "naacl": retrieve_from_naacl,
        "neurips": retrieve_from_neurips,
    }
    temp = re.compile("([a-zA-Z]+)([0-9]+)")
    confname, year = temp.match(conference).groups()
    func_retrieve = retrieval_functions[confname]
    return func_retrieve(driver, year)


def retrieve_from_neurips(driver, year):
    pdfurllist = []
    pdfnamelist = []
    abstlist = []
    title_element_list = driver.find_elements(
        By.CLASS_NAME, "paper-list"
    )  # driver.find_elements_by_class_name('ptitle')
    url_element_list = driver.find_elements(
        By.XPATH, "//a[@title='paper title']"
    )  # driver.find_elements_by_partial_link_text('pdf')
    urls = [element.get_attribute("href") for element in url_element_list]
    for i, href in tqdm(enumerate(urls)):
        # pdfnamelist.append(title_element_list[i].text)
        driver.get(href)  # Access the movie’s page
        # soup_source = BeautifulSoup(
        #    driver.page_source, "html.parser"
        # )  # Parsing content using beautifulsoup
        soup = BeautifulSoup(driver.page_source, "html.parser")
        el = soup.find_all("meta", attrs={"name": "citation_pdf_url"})
        abst_elements = driver.find_elements(By.CLASS_NAME, "container-fluid")  # soup_source.select("container-fluid")
        title = abst_elements[0].text.split("\n")[0]
        abst = abst_elements[0].text.split("Abstract\n")[-1]
        abstlist.append(abst)
        pdfnamelist.append(title)
        pdfurllist.append(el[0]["content"])
        print(title)
        # driver.get(href)
        # abstract = driver.find_element(By.ID, "abstract").text
        # abstlist.append(abstract)
        # driver.back()
        # pdfurllist.append(url_element_list[i].get_attribute("href"))
    return pdfurllist, pdfnamelist, abstlist


def retrieve_from_cvpr(driver, year):
    pdfurllist = []
    pdfnamelist = []
    abstlist = []
    title_element_list = driver.find_elements(By.CLASS_NAME, "ptitle")  # driver.find_elements_by_class_name('ptitle')
    url_element_list = driver.find_elements(By.LINK_TEXT, "pdf")  # driver.find_elements_by_partial_link_text('pdf')
    for i, element in tqdm(enumerate(url_element_list)):
        pdfnamelist.append(title_element_list[i].text)
        href = title_element_list[i].find_element(By.TAG_NAME, "a").get_attribute("href")
        driver.get(href)
        abstract = driver.find_element(By.ID, "abstract").text
        abstlist.append(abstract)
        driver.back()
        pdfurllist.append(url_element_list[i].get_attribute("href"))
    return pdfurllist, pdfnamelist, abstlist


def retrieve_from_eccv(driver, year):
    pdfurllist = []
    pdfnamelist = []
    abstlist = []
    venues = driver.find_elements(By.CLASS_NAME, "accordion")
    for venue in venues:
        if year in venue.text:
            venue.click()
    title_element_list = driver.find_elements(By.CLASS_NAME, "ptitle")  # driver.find_elements_by_class_name('ptitle')
    url_element_list = driver.find_elements(By.LINK_TEXT, "pdf")  # driver.find_elements_by_partial_link_text('pdf')
    for i, element in tqdm(enumerate(url_element_list)):
        pdfnamelist.append(title_element_list[i].text)
        href = title_element_list[i].find_element(By.TAG_NAME, "a").get_attribute("href")
        # driver.get(href)
        # abstract = driver.find_element(By.ID, "abstract").text
        # abstlist.append(abstract)
        # driver.back()
        pdfurllist.append(url_element_list[i].get_attribute("href"))
    return pdfurllist, pdfnamelist, abstlist


def retrieve_from_icml(driver, year):
    pdfurllist = []
    pdfnamelist = []
    abstlist = []
    title_element_list = driver.find_elements(By.CLASS_NAME, "title")  # driver.find_elements_by_class_name('ptitle')
    abs_list = driver.find_elements(By.LINK_TEXT, "abs")
    url_element_list = driver.find_elements(
        By.LINK_TEXT, "Download PDF"
    )  # driver.find_elements_by_partial_link_text('pdf')

    for i, element in tqdm(enumerate(url_element_list)):
        pdfnamelist.append(title_element_list[i].text)
        # href = title_element_list[i].find_element(By.TAG_NAME, "a").get_attribute('href')
        href = abs_list[i].get_attribute("href")
        driver.get(href)
        abstract = driver.find_element(By.ID, "abstract").text
        abstlist.append(abstract)
        driver.back()
        pdfurllist.append(url_element_list[i].get_attribute("href"))
    # pdb.set_trace()
    return pdfurllist, pdfnamelist, abstlist



def retrieve_from_acl(driver, year):
    
    title_element_list = driver.find_elements(By.TAG_NAME, "strong")[1:]
    url_element_list = driver.find_elements(By.LINK_TEXT, "pdf")[1:]
    title_list = [title_element.text for title_element in title_element_list]
    href_list = [title_element.find_element(By.TAG_NAME, "a").get_attribute("href") for title_element in title_element_list]

    url_element = []
    for element in url_element_list:
        pdf_name = element.get_attribute("href")
        if year == "2019" and "P19" in pdf_name:
            url_element.append(pdf_name)

    pdfurllist = []
    pdfnamelist = []
    abstlist = []
    for i, (element, title, href) in tqdm(enumerate(zip(title_element_list, title_list, href_list))):
        if ("Proceedings" in title) or ("Workshop" in title):
            continue
        if i == len(url_element_list):
            break
        pdf_name = url_element[i]
        if year == "2019" and "P19" not in pdf_name:
            continue
        try:
            driver.get(href)
            abstract = driver.find_element(By.CLASS_NAME, "acl-abstract").text
            abstlist.append(abstract)
            pdfurllist.append(pdf_name)
            pdfnamelist.append(title)
        except:
            break

        if year == "2020":
            if "main" not in pdf_name:
                break
        elif year == "2022" or year == "2021":
            if "acl-short" not in pdf_name and "acl-long" not in pdf_name:
                break
    import pdb
    pdb.set_trace()
    return pdfurllist, pdfnamelist, abstlist


def retrieve_from_emnlp(driver, year):
    pdfurllist = []
    pdfnamelist = []
    abstlist = []
    title_element_list = driver.find_elements(By.TAG_NAME, "strong")[1:]
    url_element_list = driver.find_elements(By.LINK_TEXT, "pdf")[1:]
    import pdb

    pdb.set_trace()
    for i, element in tqdm(enumerate(title_element_list)):
        if i == len(url_element_list):
            break
        pdf_name = url_element_list[i].get_attribute("href")
        if year == "2019":
            number = pdf_name.split("/")[-1].split(".pdf")[0].split("-")[-1]
            if int(number) > 1682:
                break
        if year == "2018":
            number = pdf_name.split("/")[-1].split(".pdf")[0].split("-")[-1]
            if not 1000 <= float(number) <= 1549:
                continue
        pdfnamelist.append(title_element_list[i].text)
        # href = title_element_list[i].find_element(By.TAG_NAME, "a").get_attribute('href')
        # driver.get(href)
        # abstract = driver.find_element(By.ID, "abstract").text
        # abstlist.append(abstract)
        # driver.back()
        pdfurllist.append(pdf_name)

        if year == "2020":
            if "main" not in pdf_name:
                break
        elif year == "2022" or year == "2021":
            if "main" not in pdf_name:
                break

    return pdfurllist, pdfnamelist, abstlist


def retrieve_from_naacl(driver, year):
    pdfurllist = []
    pdfnamelist = []
    abstlist = []
    title_element_list = driver.find_elements(By.TAG_NAME, "strong")[1:]
    url_element_list = driver.find_elements(By.LINK_TEXT, "pdf")[1:]
    for i, element in tqdm(enumerate(title_element_list)):
        if i == len(url_element_list):
            break
        pdf_name = url_element_list[i].get_attribute("href")
        if year == "2019":
            number = pdf_name.split("/")[-1].split(".pdf")[0].split("-")[-1]
            if not 1001 <= float(number) <= 1424:
                continue
        pdfnamelist.append(title_element_list[i].text)
        # href = title_element_list[i].find_element(By.TAG_NAME, "a").get_attribute('href')
        # driver.get(href)
        # abstract = driver.find_element(By.ID, "abstract").text
        # abstlist.append(abstract)
        # driver.back()
        pdfurllist.append(pdf_name)

        if year == "2020":
            if "main" not in pdf_name:
                break
        elif year == "2022" or year == "2021":
            if "main" not in pdf_name:
                break

    return pdfurllist, pdfnamelist, abstlist
