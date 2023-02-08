from ctypes import c_char_p
import requests
from bs4 import BeautifulSoup
import bs4
import multiprocessing
import time
import pandas as pd
import dask.dataframe as dd
from dask.diagnostics import ProgressBar


def get_meta(soup):
    try:
        if not soup.findAll("meta"):
            return 'NULL', 'NULL', 'NULL', 'NULL', 'NULL'
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

    except:
        return 'NULL', 'NULL', 'NULL', 'NULL', 'NULL'


def parse_bs(url: str, text: str, metad):
    try:
        res = requests.get(url)

        try:
            s = res.content.decode("utf-8")
            soup = BeautifulSoup(s, "html.parser")
            metad.extend(list(get_meta(soup)))
            for data in soup(['style', 'script', 'img']):
                data.decompose()
            text.value = ' '.join(soup.stripped_strings)
        except UnicodeDecodeError:
            text.value = "NULL"
            metad.extend(['NULL', 'NULL', 'NULL', 'NULL', 'NULL'])
    except:
        text.value = "NULL"
        metad.extend(list(('NULL', 'NULL', 'NULL', 'NULL', 'NULL')))


def parse_raw_texts(url):
    url = url['url_host']
    manager = multiprocessing.Manager()
    text = manager.Value(c_char_p, "NULL")
    metad = manager.list()
    p = multiprocessing.Process(target=parse_bs, name="Foo", args=(url, text, metad))
    p.start()
    time.sleep(10)
    p.terminate()
    p.join()
    if len(metad) < 5:
        metad = (["NULL"] * (5 - len(metad)))
    metad.append(str(text.value))
    columns = ['title', 'uri', 'description',
               'site_name',
               'keywords',
               'text']
    metad = pd.Series({columns[i]: metad[i] for i in range(len(columns))})
    return metad


def parse(sites_path: str, out_path: str):
    df = pd.read_csv(sites_path)
    #df=df.loc[200:204]
    df.url_host = 'http://' + df.url_host
    ddf = dd.from_pandas(df, npartitions=4)
    parse_df = {'title': 'object',
                'uri': 'object',
                'description': 'object',
                'site_name': 'object',
                'keywords': 'object',
                'text': 'object'}
    res = ddf.apply(parse_raw_texts, axis=1, meta=dd.utils.make_meta(parse_df))
    with ProgressBar():
        dft = res.compute()
    df = pd.concat([df, dft], axis=1)
    df.to_excel(out_path)
