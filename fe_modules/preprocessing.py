import polars as pl


def clean_os_type(df: pl.DataFrame, os_col: str = "cpe_model_os_type", alias: str = "cpe_model_os_type") -> pl.DataFrame:
    mapper = {
        "Android": "Android",
        "Apple iOS": "iOS",
        "iOS": "iOS",
    }
    return df.with_column(pl.col(os_col).apply(lambda x: mapper[x]).alias(alias))
