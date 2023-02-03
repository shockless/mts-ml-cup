import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


def parse_raw_texts(urls: list):
    texts = []

    for url in tqdm(urls):
        res = requests.get(url)
        html_page = res.content
        soup = BeautifulSoup(html_page, 'html.parser')
        text = soup.find_all(text=True)

        texts.append(text)

    return texts
