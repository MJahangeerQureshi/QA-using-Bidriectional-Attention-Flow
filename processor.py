# Importing Libraries
import logging
import os
import pandas as pd
import glob
import numpy as np
import torch
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
import logging
import pandas as pd
import glob
import numpy as np
import torch
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
import requests
from bs4.element import Comment
import urllib.request
from bs4 import BeautifulSoup
import wikipedia
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
import chromedriver_binary

cwd = os.path.abspath(os.path.dirname(__file__))

# Loading model and making adjustments
pd.set_option('display.max_colwidth', 500)
logger = logging.getLogger()
logger.setLevel(logging.WARNING)

archive = load_archive(
    cwd + "/bidaf-model-2017.09.15-charpad.tar.gz", cuda_device=-1)

bidaf_model = Predictor.from_archive(archive, "machine-comprehension")


def start_chrome():
    chrome_options = Options()
    chrome_options.add_argument('--window-size=1024x768')
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--single-process')
    chrome_options.add_argument('--disable-dev-shm-usage')

    
    driver = webdriver.Chrome(chrome_options=chrome_options)
    
    return driver


# Function for google search
def google_first_link(q):

    headers_Get = {
        'User-Agent':
        'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:49.0) Gecko/20100101 Firefox/49.0',
        'Accept':
        'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language':
        'en-US,en;q=0.5',
        'Accept-Encoding':
        'gzip, deflate',
        'DNT':
        '1',
        'Connection':
        'keep-alive',
        'Upgrade-Insecure-Requests':
        '1'
    }
    
    s = requests.Session()
    q = '+'.join(q.split())
    url = 'https://www.google.com/search?q=' + q + '&ie=utf-8&oe=utf-8'
    r = s.get(url, headers=headers_Get)

    soup = BeautifulSoup(r.text, "html.parser")
    output = []
    for searchWrapper in soup.find_all('h3', {
            'class': 'r'
    }):  # this line may change in future based on google's web page structure
        url = searchWrapper.find('a')["href"]
        text = searchWrapper.find('a').text.strip()
        result = {'text': text, 'url': url}
        output.append(result)

    return output


# Function for scraping visible text
def tag_visible_txt(element):
    if element.parent.name in [
            'style', 'script', 'head', 'title', 'meta', '[document]'
    ]:
        return False
    if isinstance(element, Comment):
        return False
    return True


def text_from_html(body):
    soup = BeautifulSoup(body, 'html.parser')
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible_txt, texts)
    return u" ".join(t.strip() for t in visible_texts)


def ask_google(query):

    driver = start_chrome()

    # Search for query
    query = query.replace(' ', '+')

    driver.get('http://www.google.com/search?q=' + query)

    # Get text from Google answer box

    answer = driver.execute_script(
        "return document.elementFromPoint(arguments[0], arguments[1]);", 350,
        230).text

    if answer == 'Dictionary':
        answer = driver.execute_script(
            "return document.elementFromPoint(arguments[0], arguments[1]);",
            350, 450).text

    driver.close()

    return answer


def Querry(Question):
    google_result = ask_google(Question)

    response = None
    if "https" not in google_result and "www" not in google_result and ".net" not in google_result and ".com" not in google_result and len(
            google_result) != 0 and google_result != 'map':
        response = google_result
    elif "en.wikipedia.org" in google_result:
        wiki = google_result[google_result.find("en.wikipedia.org"):
                             google_result.find("en.wikipedia.org") + 50]
        wiki = wiki.split('/')
        wiki = wiki[2]
        if '\n' in wiki:
            wiki = wiki = wiki.split('\n')[0]
        Answer_Paragraph = wikipedia.summary(wiki)
        Conversation = [{
            "passage": Answer_Paragraph,
            "question": Question,
        }]
        Precise_Answer_Paragraph = bidaf_model.predict_json(Conversation[0])

        response = Precise_Answer_Paragraph["best_span_str"]
    else:
        try:
            Answer_Paragraph = wikipedia.summary(Question)
        except:
            Answer_Paragraph = google_result

        Conversation = [{
            "passage": Answer_Paragraph,
            "question": Question,
        }]
        Precise_Answer_Paragraph = bidaf_model.predict_json(Conversation[0])

        response = Precise_Answer_Paragraph["best_span_str"]

    return response
