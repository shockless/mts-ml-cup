import pandas as pd

def fill_price(df: pd.DataFrame(), column: str = "price") -> pd.DataFrame:
    df[column] = df[column].fillna(df[column].mean())
    return df