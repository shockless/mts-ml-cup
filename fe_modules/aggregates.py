import pandas as pd
import numpy as np

from typing import Union


def get_agg_count(df: pd.DataFrame,
                  agg_col: Union[str, list] = "user_id",
                  target_col: str = None,
                  alias: str = None,
                  sort: bool = False) -> pd.DataFrame:
    if alias:
        col_name = alias
    else:
        col_name = f'{agg_col}_group_count'

    if isinstance(agg_col, str):
        agg_col = [agg_col]

    agg = df.groupby(agg_col)[target_col].count().to_frame().rename(columns={target_col: col_name}).reset_index()
    df = df.merge(agg, how="left", on=agg_col)
    if sort:
        return df.sort_values(by=agg_col)

    return df


def get_agg_sum(df: pd.DataFrame,
                agg_col: Union[str, list] = "user_id",
                target_col: str = None,
                alias: str = None,
                sort: bool = False) -> pd.DataFrame:
    if alias:
        col_name = alias
    else:
        col_name = f'{agg_col}_group_sum'

    if isinstance(agg_col, str):
        agg_col = [agg_col]

    agg = df.groupby(agg_col)[target_col].sum().to_frame().rename(columns={target_col: col_name}).reset_index()
    df = df.merge(agg, how="left", on=agg_col)
    if sort:
        return df.sort_values(by=agg_col)

    return df


def get_agg_mean(df: pd.DataFrame,
                 agg_col: Union[str, list] = "user_id",
                 target_col: str = None,
                 alias: str = None,
                 sort: bool = False) -> pd.DataFrame:
    if alias:
        col_name = alias
    else:
        col_name = f'{agg_col}_group_mean'

    if isinstance(agg_col, str):
        agg_col = [agg_col]

    agg = df.groupby(agg_col)[target_col].mean().to_frame().rename(columns={target_col: col_name}).reset_index()
    df = df.merge(agg, how="left", on=agg_col)
    if sort:
        return df.sort_values(by=agg_col)

    return df


def get_agg_max(df: pd.DataFrame,
                agg_col: Union[str, list] = "user_id",
                target_col: str = None,
                alias: str = None,
                sort: bool = False) -> pd.DataFrame:
    if alias:
        col_name = alias
    else:
        col_name = f'{agg_col}_group_max'

    if isinstance(agg_col, str):
        agg_col = [agg_col]

    agg = df.groupby(agg_col)[target_col].max().to_frame().rename(columns={target_col: col_name}).reset_index()
    df = df.merge(agg, how="left", on=agg_col)
    if sort:
        return df.sort_values(by=agg_col)

    return df


def get_agg_min(df: pd.DataFrame,
                agg_col: Union[str, list] = "user_id",
                target_col: str = None,
                alias: str = None,
                sort: bool = False) -> pd.DataFrame:
    if alias:
        col_name = alias
    else:
        col_name = f'{agg_col}_group_max'

    if isinstance(agg_col, str):
        agg_col = [agg_col]

    agg = df.groupby(agg_col)[target_col].min().to_frame().rename(columns={target_col: col_name}).reset_index()
    df = df.merge(agg, how="left", on=agg_col)
    if sort:
        return df.sort_values(by=agg_col)

    return df


def get_agg_median(df: pd.DataFrame,
                   agg_col: Union[str, list] = "user_id",
                   target_col: str = None,
                   alias: str = None,
                   sort: bool = False) -> pd.DataFrame:
    if alias:
        col_name = alias
    else:
        col_name = f'{agg_col}_group_median'

    if isinstance(agg_col, str):
        agg_col = [agg_col]

    agg = df.groupby(agg_col)[target_col].median().to_frame().rename(columns={target_col: col_name}).reset_index()
    df = df.merge(agg, how="left", on=agg_col)
    if sort:
        return df.sort_values(by=agg_col)

    return df


def get_agg_std(df: pd.DataFrame,
                agg_col: Union[str, list] = "user_id",
                target_col: str = None,
                alias: str = None,
                sort: bool = False) -> pd.DataFrame:
    if alias:
        col_name = alias
    else:
        col_name = f'{agg_col}_group_std'

    if isinstance(agg_col, str):
        agg_col = [agg_col]

    agg = df.groupby(agg_col)[target_col].std().to_frame().rename(columns={target_col: col_name}).reset_index()
    df = df.merge(agg, how="left", on=agg_col)
    if sort:
        return df.sort_values(by=agg_col)

    return df


def get_agg_nunique(df: pd.DataFrame,
                    agg_col: Union[str, list] = "user_id",
                    target_col: str = None,
                    alias: str = None,
                    sort: bool = False) -> pd.DataFrame:
    if alias:
        col_name = alias
    else:
        col_name = f'{agg_col}_group_nunique'

    if isinstance(agg_col, str):
        agg_col = [agg_col]

    agg = df.groupby(agg_col)[target_col].nunique().to_frame().rename(columns={target_col: col_name}).reset_index()
    df = df.merge(agg, how="left", on=agg_col)
    if sort:
        return df.sort_values(by=agg_col)

    return df


def get_price_of_all_cpes(df: pd.DataFrame,
                          agg_col: Union[str, list] = "user_id",
                          target_col: str = None,
                          alias: str = None,
                          sort: bool = False) -> pd.DataFrame:
    if alias:
        col_name = alias
    else:
        col_name = f'{agg_col}_group_price_of_all_cpes'

    if isinstance(agg_col, str):
        agg_col = [agg_col]

    agg = df.groupby(agg_col)[target_col].unique().sum().to_frame().rename(
        columns={target_col: col_name}).reset_index()
    df = df.merge(agg, how="left", on=agg_col)
    if sort:
        return df.sort_values(by=agg_col)

    return df


def get_top_n_mode(df: pd.DataFrame,
                   agg_col: Union[str, list] = "user_id",
                   target_col: str = None,
                   n: int = 1,
                   alias: str = None,
                   sort: bool = False) -> pd.DataFrame:
    if alias:
        col_name = alias
    else:
        col_name = f'{agg_col}_mode'

    if isinstance(agg_col, str):
        agg_col = [agg_col]

    def n_mode(s):
        temp = s.value_counts()[:n].index
        temp = np.append(temp, np.array(["<blank>"] * (n - len(temp))))
        return temp

    agg = df.groupby(agg_col)[target_col].agg(n_mode).to_frame()

    agg = pd.DataFrame(np.concatenate((np.expand_dims(agg.index.to_numpy().astype(object), axis=1),
                                       np.stack(agg[target_col].values)), axis=1),
                       columns=agg_col + [f"{col_name}_{i}" for i in range(n)])

    df = df.merge(agg, how="left", on=agg_col)
    if sort:
        return df.sort_values(by=agg_col)

    return df


def get_ratio_part_of_day(df: pd.DataFrame,
                          agg_col: Union[str, list] = "user_id") -> pd.DataFrame:
    if isinstance(agg_col, str):
        agg_col = [agg_col]

    def ratio(s):
        temp = s.value_counts()
        set_ = {"morning", "day", "evening", "night"}
        for i in set_ - set(temp.index):
            temp[i] = 0
        n = s.shape[0]
        return np.array([temp["morning"] / n, temp["day"] / n, temp["evening"] / n, temp["night"] / n])

    agg = df.groupby(agg_col)["part_of_day"].agg(ratio).to_frame()
    agg = pd.DataFrame(np.concatenate((np.expand_dims(agg.index.to_numpy().astype(object), axis=1),
                                       np.stack(agg["part_of_day"].values)), axis=1),
                       columns=agg_col + ["morning", "day", "evening", "night"])

    df = df.merge(agg, how="left", on=agg_col)
    return df
