import pandas as pd
from tqdm import tqdm

tqdm.pandas()


def get_domain(df: pd.DataFrame, url_col: str = "url_host", alias: str = "domain",
               verbose: bool = True) -> pd.DataFrame:
    splitter = lambda site: site.split(".")[-1]
    unique_urls = pd.DataFrame(df[url_col].unique(), columns=[url_col])
    if verbose:
        unique_urls[alias] = unique_urls[url_col].progress_apply(splitter)
    else:
        unique_urls[alias] = unique_urls[url_col].apply(splitter)

    return df.merge(unique_urls, on=url_col, how="left")
