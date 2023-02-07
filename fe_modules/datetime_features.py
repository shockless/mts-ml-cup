import pandas as pd
import polars as pl


def get_year(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    df["year"] = pd.DatetimeIndex(df[date_col]).year
    return df


def get_month(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    df["month"] = pd.DatetimeIndex(df[date_col]).month
    return df


def get_day(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    df["day"] = pd.DatetimeIndex(df[date_col]).day
    return df


def get_timestamp(df: pd.DataFrame, date_col: str = "date", scaler: int = 10e9, alias: str = "timestamp") -> pd.DataFrame:
    df[alias] = pd.DatetimeIndex(df[date_col]).astype(int) / scaler
    return df


def part_of_day_to_hour(df: pl.DataFrame, col: str = "part_of_day", alias: str = "hour") -> pl.DataFrame:
    mapper = {
        "morning": timedelta(hours=9),
        "day": timedelta(hours=15),
        "evening": timedelta(hours=21),
        "night": timedelta(hours=3)
    }
    
    df = df.with_column([
        pl.col(col).apply(lambda x: mapper[x]).alias(alias)
    ])
    
    return df


def add_hour_to_date(df: pl.DataFrame, date_col: str = "date", hour_col: str = "hour", alias: str = "datetime") -> pl.DataFrame: 
    return df.with_column((pl.col(date_col) + pl.col(hour_col)).alias(alias))


def get_relative_time(df: pl.DataFrame,
                      agg_col: str = "user_id",
                      sort_col: str = "timestamp",
                      date_col: str = "datetime",
                      scaler: int = 1e8) -> pl.DataFrame:
    last_dates = df.sort(sort_col).groupby(agg_col).agg(pl.col(date_col).last().alias('last_date'))
    df = df.join(last_dates, on=agg_col, how="left")
    df = df.with_column(((pl.col("last_date") - pl.col(date_col)).dt.timestamp() / scaler).alias("relative_date"))
    df = df.drop(["last_date"])

    return df
