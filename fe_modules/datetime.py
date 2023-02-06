from datetime import timedelta

import polars as pl


def get_year(df: pl.DataFrame, date_col: str = "date") -> pl.DataFrame:
    return df.with_column(pl.col(date_col).dt.year().alias("year"))


def get_month(df: pl.DataFrame, date_col: str = "date") -> pl.DataFrame:
    return df.with_column(pl.col(date_col).dt.month().alias("month"))


def get_day(df: pl.DataFrame, date_col: str = "date") -> pl.DataFrame:
    return df.with_column(pl.col(date_col).dt.day().alias("day"))


def get_timestamp(df: pl.DataFrame, date_col: str = "date", scaler: int = 1e8) -> pl.DataFrame:
    return df.with_column((pl.col(date_col).dt.timestamp() /  scaler).alias("timestamp"))


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
