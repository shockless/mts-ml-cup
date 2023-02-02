import polars as pl


def get_year(df: pl.DataFrame, date_col: str) -> pl.DataFrame:
    return df.with_column(pl.col(date_col).dt.year().alias("year"))


def get_month(df: pl.DataFrame, date_col: str) -> pl.DataFrame:
    return df.with_column(pl.col(date_col).dt.month().alias("month"))


def get_day(df: pl.DataFrame, date_col: str) -> pl.DataFrame:
    return df.with_column(pl.col(date_col).dt.day().alias("day"))


def get_timestamp(df: pl.DataFrame, date_col: str) -> pl.DataFrame:
    return df.with_column(pl.col(date_col).dt.timestamp().alias("timestamp"))


def get_relative_time(df: pl.DataFrame,
                      agg_col: str = "user_id",
                      sort_col: str = "timestamp",
                      date_col: str = "date") -> pl.DataFrame:
    last_dates = df.sort(sort_col).groupby(agg_col).agg(pl.col(date_col).last().alias('last_date'))
    df = df.join(last_dates, on=agg_col, how="left")
    df = df.with_column((pl.col("last_date") - pl.col(date_col)).alias("relative_date"))
    df = df.drop(["last_date"])

    return df
