from ctypes import c_char_p
import requests
from bs4 import BeautifulSoup
import bs4
import multiprocessing
import time
import pandas as pd
import dask.dataframe as dd
from dask.diagnostics import ProgressBar


def parse_bs(url: str, text: str):
    try:
        res = requests.get(url)
        try:
            s = res.content.decode("utf-8")
            soup = BeautifulSoup(s, "html.parser")
            for data in soup(['style', 'script', 'img']):
                data.decompose()
            text.value = ' '.join(soup.stripped_strings)
        except UnicodeDecodeError:
            text.value = "NULL"
    except:
        text.value = "NULL"


def parse_raw_texts(url):
    url = url['url_host']
    manager = multiprocessing.Manager()
    lock = manager.Lock()
    text = manager.Value(c_char_p, "NULL")
    p = multiprocessing.Process(target=parse_bs, name="Foo", args=(url, text,))
    p.start()
    time.sleep(10)
    p.terminate()
    p.join()
    return str(text.value)


def parse(sites_path: str, out_path: str):
    df = pd.read_csv(sites_path)
    df['text'] = [''] * df.shape[0]
    df.url_host = 'http://' + df.url_host
    ddf = dd.from_pandas(df, npartitions=4)
    res = ddf.apply(parse_raw_texts, axis=1, meta=pd.Series(dtype='object', name='text'))
    with ProgressBar():
        dft = res.compute()

    df['text'] = dft
    df.to_csv(out_path)
