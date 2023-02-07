from ctypes import c_char_p
import requests
from bs4 import BeautifulSoup
import multiprocessing
import time
import pandas as pd
import dask.dataframe as dd
from dask.diagnostics import ProgressBar


def parse_bs(url: str, text: str):
    try:
        res = requests.get(url)
        soup = BeautifulSoup(res.content, "html.parser")
        for data in soup(['style', 'script']):
            data.decompose()
        text.value = ' '.join(soup.stripped_strings)
    except requests.exceptions.ConnectionError:
        text.value = "NULL"


def parse_raw_texts(url):
    url = url['url_host']
    manager = multiprocessing.Manager()
    text = manager.Value(c_char_p, "NULL")
    text.value = "NULL"
    p = multiprocessing.Process(target=parse_bs, name="Foo", args=(url, text,))
    p.start()
    time.sleep(1)
    p.terminate()
    p.join()

    return text.value


def parse(sites_path: str, out_path: str):
    df = pd.read_csv(sites_path)
    df.url_host = 'http://' + df.url_host
    df['text'] = [''] * df.shape[0]
    ddf = dd.from_pandas(df, npartitions=8)
    res = ddf.apply(parse_raw_texts, axis=1, meta=('text', 'object'))
    with ProgressBar():
        res.compute()

    df.to_csv(out_path)


