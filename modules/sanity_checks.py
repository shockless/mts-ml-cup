import warnings

import pandas as pd


def null_check(df: pd.DataFrame) -> None:
    missing = df.isnull().sum()
    if missing.sum() > 0:
        warnings.warn(f"""
                Warning! Detected null column values {dict(zip(df.columns, missing))}
            """)
