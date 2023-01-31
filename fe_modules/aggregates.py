import polars as pl


def get_agg_count(df: pl.DataFrame, agg_col: str = "user_id", sort: bool = True) -> pl.DataFrame:
    agg = df.groupby(agg_col).agg(pl.col("date").count().alias(f'{agg_col}_group_count'))

    if sort:
        return agg.sort(agg_col)

    return agg

def get_agg_sum(df: pl.DataFrame, target_col: str, agg_col: str = "user_id", sort: bool = True) -> pl.DataFrame:
    agg = df.groupby(agg_col).agg(pl.col(target_col).sum().alias(f'{agg_col}_group_{target_col}_sum'))

    if sort:
        return agg.sort(agg_col)

    return agg


def get_agg_mean(df: pl.DataFrame, target_col: str, agg_col: str = "user_id", sort: bool = True) -> pl.DataFrame:
    agg = df.groupby(agg_col).agg(pl.col(target_col).mean().alias(f'{agg_col}_group_{target_col}_mean'))

    if sort:
        return agg.sort(agg_col)

    return agg


def get_agg_mode(df: pl.DataFrame, target_col: str, agg_col: str = "user_id", sort: bool = True) -> pl.DataFrame:
    agg = df.groupby(agg_col).agg(pl.col(target_col).mode().apply(lambda x: x[0]).alias(f'{agg_col}_group_{target_col}_mean'))

    if sort:
        return agg.sort(agg_col)

    return agg


def get_agg_median(df: pl.DataFrame, target_col: str, agg_col: str = "user_id", sort: bool = True) -> pl.DataFrame:
    agg = df.groupby(agg_col).agg(pl.col(target_col).median().alias(f'{agg_col}_group_{target_col}_median'))

    if sort:
        return agg.sort(agg_col)

    return agg


def get_agg_std(df: pl.DataFrame, target_col: str, agg_col: str = "user_id", sort: bool = True) -> pl.DataFrame:
    agg = df.groupby(agg_col).agg(pl.col(target_col).std().alias(f'{agg_col}_group_{target_col}_std'))

    if sort:
        return agg.sort(agg_col)

    return agg


def get_agg_n_unique(df: pl.DataFrame, target_col: str, agg_col: str = "user_id", sort: bool = True) -> pl.DataFrame:
    agg = df.groupby(agg_col).agg(pl.col(target_col).n_unique().alias(f'{agg_col}_group_{target_col}_n_unique'))

    if sort:
        return agg.sort(agg_col)

    return agg
