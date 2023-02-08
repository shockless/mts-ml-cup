import pandas as pd


def get_agg_count(df: pd.DataFrame,
                  agg_col: str = "user_id",
                  target_col: str = None,
                  col: str = None,
                  sort: bool = False) -> pd.DataFrame:
    if col:
        col_name = col
    else:
        col_name = f'{agg_col}_group_count'

    agg = df.groupby([agg_col])[target_col].count().to_frame().rename(columns={target_col: col_name}).reset_index()
    df = df.merge(agg, how="left", on=[agg_col])
    if sort:
        return df.sort_values(by=agg_col)

    return df


def get_agg_sum(df: pd.DataFrame,
                agg_col: str = "user_id",
                target_col: str = None,
                col: str = None,
                sort: bool = False) -> pd.DataFrame:
    if col:
        col_name = col
    else:
        col_name = f'{agg_col}_group_sum'

    agg = df.groupby([agg_col])[target_col].sum().to_frame().rename(columns={target_col: col_name}).reset_index()
    df = df.merge(agg, how="left", on=[agg_col])
    if sort:
        return df.sort_values(by=agg_col)

    return df


def get_agg_mode(df: pd.DataFrame,
                 agg_col: str = "user_id",
                 target_col: str = None,
                 col: str = None,
                 sort: bool = False) -> pd.DataFrame:
    if col:
        col_name = col
    else:
        col_name = f'{agg_col}_group_mode'

    agg = df.groupby([agg_col])[target_col].mode().to_frame().rename(columns={target_col: col_name}).reset_index()
    df = df.merge(agg, how="left", on=[agg_col])
    if sort:
        return df.sort_values(by=agg_col)

    return df


def get_agg_mean(df: pd.DataFrame,
                 agg_col: str = "user_id",
                 target_col: str = None,
                 col: str = None,
                 sort: bool = False) -> pd.DataFrame:
    if col:
        col_name = col
    else:
        col_name = f'{agg_col}_group_mean'

    agg = df.groupby([agg_col])[target_col].mean().to_frame().rename(columns={target_col: col_name}).reset_index()
    df = df.merge(agg, how="left", on=[agg_col])
    if sort:
        return df.sort_values(by=agg_col)

    return df


def get_agg_max(df: pd.DataFrame,
                agg_col: str = "user_id",
                target_col: str = None,
                col: str = None,
                sort: bool = False) -> pd.DataFrame:
    if col:
        col_name = col
    else:
        col_name = f'{agg_col}_group_max'

    agg = df.groupby([agg_col])[target_col].max().to_frame().rename(columns={target_col: col_name}).reset_index()
    df = df.merge(agg, how="left", on=[agg_col])
    if sort:
        return df.sort_values(by=agg_col)

    return df


def get_agg_median(df: pd.DataFrame,
                   agg_col: str = "user_id",
                   target_col: str = None,
                   col: str = None,
                   sort: bool = False) -> pd.DataFrame:
    if col:
        col_name = col
    else:
        col_name = f'{agg_col}_group_median'

    agg = df.groupby([agg_col])[target_col].median().to_frame().rename(columns={target_col: col_name}).reset_index()
    df = df.merge(agg, how="left", on=[agg_col])
    if sort:
        return df.sort_values(by=agg_col)

    return df


def get_agg_std(df: pd.DataFrame,
                agg_col: str = "user_id",
                target_col: str = None,
                col: str = None,
                sort: bool = False) -> pd.DataFrame:
    if col:
        col_name = col
    else:
        col_name = f'{agg_col}_group_std'

    agg = df.groupby([agg_col])[target_col].std().to_frame().rename(columns={target_col: col_name}).reset_index()
    df = df.merge(agg, how="left", on=[agg_col])
    if sort:
        return df.sort_values(by=agg_col)

    return df


def get_agg_nunique(df: pd.DataFrame,
                    agg_col: str = "user_id",
                    target_col: str = None,
                    col: str = None,
                    sort: bool = False) -> pd.DataFrame:
    if col:
        col_name = col
    else:
        col_name = f'{agg_col}_group_nunique'

    agg = df.groupby([agg_col])[target_col].nunique().to_frame().rename(columns={target_col: col_name}).reset_index()
    df = df.merge(agg, how="left", on=[agg_col])
    if sort:
        return df.sort_values(by=agg_col)

    return df


def get_price_of_all_cpes(df: pd.DataFrame,
                          agg_col: str = "user_id",
                          target_col: str = None,
                          col: str = None,
                          sort: bool = False) -> pd.DataFrame:
    if col:
        col_name = col
    else:
        col_name = f'{agg_col}_group_price_of_all_cpes'

    agg = df.groupby([agg_col])[target_col].unique().sum().to_frame().rename(
        columns={target_col: col_name}).reset_index()
    df = df.merge(agg, how="left", on=[agg_col])
    if sort:
        return df.sort_values(by=agg_col)

    return df
