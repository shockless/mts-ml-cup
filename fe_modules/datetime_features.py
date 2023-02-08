import pandas as pd
from pandas import Timedelta


def get_year(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    df["year"] = pd.DatetimeIndex(df[date_col]).year
    return df


def get_month(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    df["month"] = pd.DatetimeIndex(df[date_col]).month
    return df


def get_day(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    df["day"] = pd.DatetimeIndex(df[date_col]).day
    return df


def get_timestamp(df: pd.DataFrame, date_col: str = "date", alias: str = "timestamp",
                  scaler: int = 10e9) -> pd.DataFrame:
    df[alias] = pd.DatetimeIndex(df[date_col]).astype(int) / scaler
    return df


def get_day_of_year(df: pd.DataFrame, date_col: str = "date", alias: str = "day_of_year") -> pd.DataFrame:
    df[alias] = pd.DatetimeIndex(df[date_col]).day_of_year
    return df


def get_day_of_week(df: pd.DataFrame, date_col: str = "date", alias: str = "day_of_week") -> pd.DataFrame:
    df[alias] = pd.DatetimeIndex(df[date_col]).day_of_week
    return df


def get_holiday_name(df: pd.DataFrame, date_col: str = "date", alias: str = "holiday") -> pd.DataFrame:
    mapper = dict()
    for i in range(8):
        mapper[f"1-{i}"] = "Новогодние каникулы"
    mapper[f"1-7"] = "Рождество Христово"
    mapper[f"2-23"] = "День защитника Отечества"
    mapper[f"3-8"] = "Международный женский день"
    mapper[f"5-1"] = "Праздник Весны и Труда"
    mapper[f"5-9"] = "День Победы"
    mapper[f"6-12"] = "День России"
    mapper[f"11-4"] = "День народного единства"

    def fill_func(x):
        if f"{x.month}-{x.day}" in mapper:
            return mapper[f"{x.month}-{x.day}"]
        else:
            return "Не праздник"

    df[alias] = df[date_col].apply(fill_func)
    return df


def part_of_day_to_hour(df: pd.DataFrame, col: str = "part_of_day", alias: str = "hour") -> pd.DataFrame:
    mapper = {
        "morning": Timedelta(hours=9),
        "day": Timedelta(hours=15),
        "evening": Timedelta(hours=21),
        "night": Timedelta(hours=3)
    }
    df[alias] = df[col].map(mapper)
    return df


def add_hour_to_date(df: pd.DataFrame, date_col: str = "date", hour_col: str = "hour",
                     alias: str = "datetime") -> pd.DataFrame:
    df[date_col] = pd.to_datetime(df[date_col])
    df[alias] = df[date_col] + df[hour_col]
    return df


def get_relative_time(df: pd.DataFrame,
                      agg_col: str = "user_id",
                      date_col: str = "datetime",
                      alias: str = "relative_date",
                      return_dtype: str = "timedelta",
                      scaler: int = 100) -> pd.DataFrame:
    last_dates = pd.DataFrame(df.groupby([agg_col])[date_col].max()).rename(columns={date_col: "last_date"})
    df = df.merge(last_dates, on=agg_col, how="left")
    if return_dtype == "timedelta":
        df[alias] = df["last_date"] - df[date_col]
    elif return_dtype == "timestamp":
        df[alias] = (df["last_date"] - df[date_col]).dt.total_seconds() / scaler
    df = df.drop(["last_date"], axis=1)
    return df
