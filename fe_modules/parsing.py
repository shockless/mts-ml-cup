from ctypes import c_char_p
import random
import requests
from bs4 import BeautifulSoup
import bs4
import multiprocessing
import time
import pandas as pd
import dask.dataframe as dd
from dask.diagnostics import ProgressBar

import warnings
warnings.filterwarnings("ignore")
def get_session(proxy):
    session = requests.Session()
    session.proxies = {"http": proxy, "https": proxy}
    return session


def get_site_name(row):
    site = str(row).split('.')
    site[0] = site[0][7:]
    if (len(site) >= 2):
        site = site[-2] + '.' + site[-1]
    else:
        site = row
    return site


def get_meta(soup):
    try:
        if not soup.findAll("meta"):
            return soup.title.text, 'NULL', 'NULL', 'NULL', 'NULL'
        uri = soup.find("meta", property="og:url")
        if not uri:
            uri = 'NULL'
        else:
            uri = uri["content"]

        title = soup.find("meta", property="og:title")
        title = title["content"] if title else soup.title.text
        if not title:
            title = 'NULL'

        description = soup.find("meta", property="og:description")
        if not description:
            description = 'NULL'
        else:
            description = description["content"]

        site_name = soup.find("meta", property="og:site_name")
        if not site_name:
            site_name = 'NULL'
        else:
            site_name = site_name["content"]

        keywords = soup.find("meta", {"name": "keywords"})
        if not keywords:
            keywords = 'NULL'
        else:
            keywords = keywords["content"]
        return title, uri, description, site_name, keywords

    except Exception as e:
        return 'NULL', 'NULL', 'NULL', 'NULL', 'NULL'


def get_content(res):
    try:
        s = res.content.decode("utf-8")
        soup = BeautifulSoup(s, "html.parser")
        metad = list(get_meta(soup))
        for data in soup(['style', 'script', 'img']):
            data.decompose()

        text = ''.join(soup.stripped_strings)
    except UnicodeDecodeError as e:
        text = "NULL"
        metad = ['NULL', 'NULL', 'NULL', 'NULL', 'NULL']
    return [text, metad]


def get_free_proxies():
    url = "http://free-proxy-list.net/"
    soup = BeautifulSoup(requests.get(url).content, "html.parser")
    proxies = []
    for row in soup.find_all("table")[0].find_all("tr")[1:]:
        tds = row.find_all("td")
        try:
            if tds[4].text.strip() == 'elite proxy' and tds[6].text.strip() == 'yes' and tds[5].text.strip() == 'yes':
                ip = tds[0].text.strip()
                port = tds[1].text.strip()
                host = f"{ip}:{port}"
                proxies.append(host)
        except IndexError:
            continue
    return proxies


class parser:

    def __init__(self):
        self.proxy = random.choice(get_free_proxies())

    def parse_bs(self, url: str, text, metad, timeout):

        while True:
            try:
                s = get_session(self.proxy)
                res = s.get(url, timeout=timeout)
                try:
                    res = get_content(res)

                    text = res[0]
                    metad = res[1]
                    if '403 Forbidden' in text or text == '' or metad[0] == '403 Forbidden':
                        url = get_site_name(url)
                        res = s.get(url, timeout=timeout)
                        res = get_content(res)
                        text = res[0]
                        metad = res[1]

                    elif 'Yandex SmartCaptcha' in text:
                        text = "NULL"
                        metad = ['NULL', 'NULL', 'NULL', 'NULL', 'NULL']

                    if text == '':
                        text = "NULL"

                except Exception as e:
                    pass
                break

            except Exception as e:
                self.proxy = random.choice(get_free_proxies())
                if e == TypeError:
                    text = "NULL"
                    metad = ['NULL', 'NULL', 'NULL', 'NULL', 'NULL']
                    break

        s.close()
        return text, metad

    def parse_raw_texts(self, url, timeout):
        url = url['url_host']
        text = "NULL"
        metad = []
        text, metad = self.parse_bs(url, text, metad, timeout)
        if len(metad) < 5:
            metad = (["NULL"] * (5 - len(metad)))
        metad.append(str(text)[:32760])
        columns = ['title', 'uri', 'description',
                   'site_name',
                   'keywords',
                   'text']
        # print(metad)
        metad = pd.Series({columns[i]: metad[i] for i in range(len(columns))})
        return metad

    def parse(self, sites_path: str, out_path: str, timeout: int):
        df = pd.read_csv(sites_path)
        #   df = df.loc[200:204]
        df.url_host = df.url_host
        ddf = dd.from_pandas(df, npartitions=4)
        parse_df = {'title': 'object',
                    'uri': 'object',
                    'description': 'object',
                    'site_name': 'object',
                    'keywords': 'object',
                    'text': 'object'}
        res = ddf.apply(self.parse_raw_texts, axis=1, meta=dd.utils.make_meta(parse_df), args=(timeout,))
        with ProgressBar():
            dft = res.compute()
        #df = pd.concat([df, dft], axis=1)
        dft.to_excel(out_path)
