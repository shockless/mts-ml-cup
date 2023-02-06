import numpy as np
import polars as pl


def generate_cyclical_features(df: pl.DataFrame, col_name: str, period: int, start_num: int = 0) -> pl.DataFrame:
    sin_generator = lambda x: np.sin(2 * np.pi * (df[col_name] - start_num) / period)
    cos_generator = lambda x: np.cos(2 * np.pi * (df[col_name] - start_num) / period)
    df = df.with_column(pl.col(col_name).apply(sin_generator).alias(f"sin_{col_name}"))
    df = df.with_column(pl.col(col_name).apply(cos_generator).alias(f"cos_{col_name}"))
    return df
