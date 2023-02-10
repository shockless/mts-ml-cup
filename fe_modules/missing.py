import pandas as pd

def fill_price(df: pd.DataFrame(), column: str = "price") -> pd.DataFrame:
    df[column] = df[column].fillna(df[column].mean())
    return df

def map_prices(df: pd.DataFrame(), price_path: str) -> pd.DataFrame:
    price = pd.read_csv('phones_with_prices.csv', usecols=['cpe_manufacturer_name','cpe_model_name','price'])
    df = df.merge(price[['price']], on="cpe_model_name", how="left")
    return df
