import polars as pl


def get_agg_sum(df: pl.DataFrame, target_col: str, agg_col: str = "user_id") -> pl.DataFrame:
    return df.groupby(agg_col).agg(pl.col(target_col).sum().alias(f'{agg_col}_sum'))


def get_agg_mean(df: pl.DataFrame, target_col: str, agg_col: str = "user_id") -> pl.DataFrame:
    return df.groupby(agg_col).agg(pl.col(target_col).mean().alias(f'{agg_col}_mean'))


def get_agg_mode(df: pl.DataFrame, target_col: str, agg_col: str = "user_id") -> pl.DataFrame:
    return df.groupby(agg_col).agg(pl.col(target_col).mode().apply(lambda x: x[0]).alias(f'{agg_col}_mean'))


def get_agg_median(df: pl.DataFrame, target_col: str, agg_col: str = "user_id") -> pl.DataFrame:
    return df.groupby(agg_col).agg(pl.col(target_col).median().alias(f'{agg_col}_median'))


def get_agg_std(df: pl.DataFrame, target_col: str, agg_col: str = "user_id") -> pl.DataFrame:
    return df.groupby(agg_col).agg(pl.col(target_col).std().alias(f'{agg_col}_std'))


def get_agg_n_unique(df: pl.DataFrame, target_col: str, agg_col: str = "user_id") -> pl.DataFrame:
    return df.groupby(agg_col).agg(pl.col(target_col).n_unique().alias(f'{target_col}_n_unique'))


