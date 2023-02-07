import polars as pl


def get_agg_count(df: pl.DataFrame,
                  agg_col: str = "user_id",
                  alias: str = None,
                  sort: bool = True) -> pl.DataFrame:
    if alias:
        alias_name = alias
    else:
        alias_name = f'{agg_col}_group_count'

    agg = df.groupby(agg_col).agg(pl.col("date").count().alias(alias_name))

    df = df.join(agg, on=agg_col, how="left")

    if sort:
        return df.sort(agg_col)

    return df


def get_agg_sum(df: pl.DataFrame,
                target_col: str,
                agg_col: str = "user_id",
                alias: str = None,
                sort: bool = True) -> pl.DataFrame:
    if alias:
        alias_name = alias
    else:
        alias_name = f'{agg_col}_group_{target_col}_sum'

    agg = df.groupby(agg_col).agg(pl.col(target_col).sum().alias(alias_name))

    df = df.join(agg, on=agg_col, how="left")

    if sort:
        return df.sort(agg_col)

    return df

  
def get_agg_mean(df: pl.DataFrame,
                 target_col: str,
                 agg_col: str = "user_id",
                 alias: str = None,
                 sort: bool = True) -> pl.DataFrame:
    if alias:
        alias_name = alias
    else:
        alias_name = f'{agg_col}_group_{target_col}_mean'

    agg = df.groupby(agg_col).agg(pl.col(target_col).mean().alias(alias_name))

    df = df.join(agg, on=agg_col, how="left")

    if sort:
        return df.sort(agg_col)

    return df

  
def get_agg_mode(df: pl.DataFrame,
                 target_col: str,
                 agg_col: str = "user_id",
                 alias: str = None,
                 sort: bool = True) -> pl.DataFrame:
    if alias:
        alias_name = alias
    else:
        alias_name = f'{agg_col}_group_{target_col}_mode'

    agg = df.groupby(agg_col).agg(pl.col(target_col).mode().apply(lambda x: x[0]).alias(alias_name))

    df = df.join(agg, on=agg_col, how="left")

    if sort:
        return df.sort(agg_col)
      
    return df

def get_agg_max(df: pl.DataFrame,
                 target_col: str,
                 agg_col: str = "user_id",
                 alias: str = None,
                 sort: bool = True) -> pl.DataFrame:
    if alias:
        alias_name = alias
    else:
        alias_name = f'{agg_col}_group_{target_col}_max'

    agg = df.groupby(agg_col).agg(pl.col(target_col).max().alias(alias_name))

    df = df.join(agg, on=agg_col, how="left")

    if sort:
        return df.sort(agg_col)

    return df
  
  
def get_agg_min(df: pl.DataFrame,
                 target_col: str,
                 agg_col: str = "user_id",
                 alias: str = None,
                 sort: bool = True) -> pl.DataFrame:
    if alias:
        alias_name = alias
    else:
        alias_name = f'{agg_col}_group_{target_col}_min'

    agg = df.groupby(agg_col).agg(pl.col(target_col).min().alias(alias_name))

    df = df.join(agg, on=agg_col, how="left")

    if sort:
        return df.sort(agg_col)

    return df

  
def get_agg_median(df: pl.DataFrame,
                   target_col: str,
                   agg_col: str = "user_id",
                   alias: str = None,
                   sort: bool = True) -> pl.DataFrame:
    if alias:
        alias_name = alias
    else:
        alias_name = f'{agg_col}_group_{target_col}_median'

    agg = df.groupby(agg_col).agg(pl.col(target_col).median().alias(alias_name))

    df = df.join(agg, on=agg_col, how="left")

    if sort:
        return df.sort(agg_col)

    return df


def get_agg_std(df: pl.DataFrame,
                target_col: str,
                agg_col: str = "user_id",
                alias: str = None,
                sort: bool = True) -> pl.DataFrame:
    if alias:
        alias_name = alias
    else:
        alias_name = f'{agg_col}_group_{target_col}_std'

    agg = df.groupby(agg_col).agg(pl.col(target_col).std().alias(alias_name))

    df = df.join(agg, on=agg_col, how="left")

    if sort:
        return df.sort(agg_col)

    return df


def get_agg_n_unique(df: pl.DataFrame,
                     target_col: str,
                     agg_col: str = "user_id",
                     alias: str = None,
                     sort: bool = True) -> pl.DataFrame:
    if alias:
        alias_name = alias
    else:
        alias_name = f'{agg_col}_group_{target_col}_n_unique'

    agg = df.groupby(agg_col).agg(pl.col(target_col).n_unique().alias(alias_name))

    df = df.join(agg, on=agg_col, how="left")

    if sort:
        return df.sort(agg_col)

    return df


def get_price_of_all_cpes(df: pl.DataFrame,
                          agg_col: str = "user_id",
                          price_col: str = "price",
                          alias: str = "total_price",
                          sort: bool = True) -> pl.DataFrame:
    if alias:
        alias_name = alias
    else:
        alias_name = f'total_price'

    agg = df.groupby(agg_col).agg(pl.col(price_col).unique().sum().alias(alias_name))

    df = df.join(agg, on=agg_col, how="left")

    if sort:
        return df.sort(agg_col)

    return df
