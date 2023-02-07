import pandas as pd


def clean_os_type(df: pd.DataFrame, os_col: str = "cpe_model_os_type", alias: str = "cpe_model_os_type") -> pd.DataFrame:
    mapper = {
        "Android": "Android",
        "Apple iOS": "iOS",
        "iOS": "iOS",
    }
    df[alias] = df[os_col].map(mapper)
    return df
