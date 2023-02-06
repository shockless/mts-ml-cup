import polars as pl


def get_domain(df, url_col: str = "url_host", alias: str = "domain"):
    splitter = lambda site: site.split(".")[-1]
    unique_urls = pl.DataFrame(df[url_col].unique())
    unique_urls = unique_urls.with_column(pl.col(url_col).apply(splitter).alias(alias))

    return df.join(unique_urls, on="url_host",how="left")

