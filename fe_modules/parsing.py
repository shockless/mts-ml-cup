import random
import requests
from bs4 import BeautifulSoup
import pandas as pd
import dask.dataframe as dd
from dask.diagnostics import ProgressBar

import warnings

warnings.filterwarnings("ignore")


def get_session(proxy):
    session = requests.Session()
    if proxy:
        session.proxies = {"http": proxy, "https": proxy}
    return session


def get_site_name(row):
    site = str(row).split('.')
    site[0] = site[0][7:]
    if len(site) > 2:
        site = 'http://' + site[-2] + '.' + site[-1]
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
        try:
            enc = res.encoding
            if enc:
                if enc == 'Win1251' or enc == 'win-1251':
                    enc = 'cp1251'
                s = res.content.decode(enc, errors='replace').encode('utf-8', errors='replace')
            else:
                s = res.content.decode('utf-8', errors='replace').encode('utf-8', errors='replace')
        except UnicodeDecodeError as e:
            print(enc)
            try:
                enc = res.encoding
                if enc:
                    s = res.text.decode(enc, errors='replace')
                else:
                    s = res.text.decode('utf-8', errors='replace')
            except:
                raise e

        soup = BeautifulSoup(s, "html.parser")
        metad = list(get_meta(soup))
        for data in soup(['style', 'script', 'img']):
            data.decompose()

        text = str(''.join(soup.stripped_strings))
    except requests.exceptions.ContentDecodingError as e:
        print(e)
        text = "NULL"
        metad = ['NULL', 'NULL', 'NULL', 'NULL', 'NULL']
    except:
        raise e
    return [text, metad]


def get_free_proxies():
    url = "http://free-proxy-list.net/"
    soup = BeautifulSoup(requests.get(url).content, "html.parser")
    proxies = []
    for row in soup.find_all("table")[0].find_all("tr")[1:]:
        tds = row.find_all("td")
        try:
            if tds[4].text.strip() == 'elite proxy' and tds[6].text.strip() == 'yes':
                ip = tds[0].text.strip()
                port = tds[1].text.strip()
                host = f"{ip}:{port}"
                proxies.append(host)
        except IndexError:
            continue
    return proxies


def get_headers():
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36',
        'Mozilla/5.0 (iPhone; CPU iPhone OS 12_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148',
        'Mozilla/5.0 (Linux; Android 11; SM-G960U) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.72 Mobile Safari/537.36'
    ]
    user_agent = random.choice(user_agents)
    headers = {'User-Agent': user_agent}
    return headers


def get_content_url(url, proxy=None, timeout=5, verify=False, headers={}):
    s = get_session(proxy)
    res = s.get(url, timeout=timeout, verify=verify, headers=headers)
    res = get_content(res)
    try:
        s.delete(url=url, headers=headers, timeout=timeout)
    except:
        pass
    s.close()
    return res[0], res[1]


class parser:

    def __init__(self):
        self.proxy = random.choice(get_free_proxies())

    def parse_bs(self, url: str, text, metad, timeout, max_retries):
        if len(str(url).split('.')) > 1:
            try:
                text, metad = get_content_url(url, None, timeout, headers=get_headers())

                if '403 Forbidden' in text or text == '' or metad[0] == '403 Forbidden':

                    try:
                        url = get_site_name(url)
                        text, metad = get_content_url(url, None, timeout)

                    except Exception as e:
                        raise e

                elif 'Yandex SmartCaptcha' in text:
                    text = "NULL"
                    metad = ['NULL', 'NULL', 'NULL', 'NULL', 'NULL']

            except Exception as e:

                if isinstance(e, requests.exceptions.InvalidURL) or \
                        isinstance(e, requests.exceptions.ReadTimeout) or \
                        isinstance(e, requests.exceptions.TooManyRedirects) or \
                        isinstance(e, requests.exceptions.ConnectTimeout) or \
                        isinstance(e, requests.exceptions.ConnectionError) or \
                        isinstance(e, requests.exceptions.ContentDecodingError):
                    try:

                        text, metad = get_content_url(url, None, timeout, headers=get_headers())
                    except:
                        print(e)
                        text = "NULL"
                        metad = ['NULL', 'NULL', 'NULL', 'NULL', 'NULL']
                else:
                    print(url)
                    raise e
        print(url)
        return text, metad

    def parse_raw_texts(self, url, timeout, max_retries):
        url = url['url_host']
        text = "NULL"
        metad = []
        text, metad = self.parse_bs(url, text, metad, timeout, max_retries)
        if len(metad) < 5:
            metad = (["NULL"] * (5 - len(metad)))
        metad.append(str(text)[:32760])
        columns = ['url', 'title', 'uri', 'description',
                   'site_name',
                   'keywords',
                   'text']
        # print(metad)
        metad.insert(0, url)
        metad = pd.Series({columns[i]: metad[i] for i in range(len(columns))})
        return metad

    def parse(self, sites_path: str, out_path: str, n_partitions: int, timeout: int, max_retries: int = 3, start=None,
              end=None):
        df = pd.read_csv(sites_path)
        if not end:
            end = df.shape[0] - 1
        if not start:
            start = 0
        df = df.iloc[start:end]
        df.url_host = df.url_host
        ddf = dd.from_pandas(df, npartitions=n_partitions)
        parse_df = {'url': 'object',
                    'title': 'object',
                    'uri': 'object',
                    'description': 'object',
                    'site_name': 'object',
                    'keywords': 'object',
                    'text': 'object'}
        res = ddf.apply(self.parse_raw_texts, axis=1, meta=dd.utils.make_meta(parse_df), args=(timeout, max_retries,))
        with ProgressBar():
            dft = res.compute()
        # df = pd.concat([df, dft], axis=1)
        dft.to_excel(out_path)
