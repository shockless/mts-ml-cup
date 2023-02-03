import warnings

import polars as pl


def null_check(df: pl.DataFrame) -> None:
    if df.null_count().to_numpy()[0].sum() > 0:
        warnings.warn(f"""
                Warning! Detected null column values {dict(zip(df.columns, df.null_count().to_numpy()[0]))}
            """)
