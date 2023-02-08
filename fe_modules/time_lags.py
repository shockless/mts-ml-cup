import pandas as pd


def generate_time_lags(df: pd.DataFrame, agg_column: str = "user_id", time_column: str = "datetime",
                       shift_column: str = "url_host", n_lags: int = 1) -> pd.DataFrame:
    df = df.sort_values(
        by=[agg_column, time_column], ascending=[True, True]
    )
    for i in range(1, n_lags + 1):
        df["match_id"] = df["user_id"].eq(df["user_id"].shift(i))
        df[f"lag_{shift_column}_{i}"] = df[shift_column].shift(i)
        df.loc[~df["match_id"], f"lag_{shift_column}_{i}"] = None
        df = df.drop(["match_id"], axis=1)
    return df
