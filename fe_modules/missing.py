import pandas as pd


def map_prices(df: pd.DataFrame(), folder_path: str = "../external_data", price_path: str = 'phones_with_prices.csv') -> pd.DataFrame:
    price = pd.read_csv(f"{folder_path}/{price_path}", usecols=['cpe_manufacturer_name', 'cpe_model_name', 'price'])
    df = df.merge(price[['cpe_model_name', 'price']].rename(columns={"price": "missing_price"}),
                  on="cpe_model_name",
                  how="left")
    df["price"][df["price"].isnull()] = df["missing_price"]
    del df["missing_price"]
    return df
