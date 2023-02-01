import polars as pl


def get_domain(df, url_col: str = "url_host"):
    splitter = lambda site: site.split(".")[-1]
    
    return df.with_column(pl.col(url_col).apply(splitter).alias("domain"))