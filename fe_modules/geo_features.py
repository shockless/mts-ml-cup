from aggregates import get_agg_mean, get_agg_min, get_agg_max
import pndas as pd


def mean_first_visit(df: pl.DataFrame) -> pl.DataFrame:
    first_visit = get_agg_min(df, agg_col=['user_id', 'date'], alias='first_visit')
    mean_fv = get_agg_mean(first_visit, 'mean_fv')
    del mean_fv['first_visit']
    return mean_fv


def mean_last_visit(df: pl.DataFrame) -> pl.DataFrame:
    last_visit = get_agg_max(df, agg_col=['user_id', 'date'], alias='last_visit')
    mean_lv = get_agg_mean(last_visit, 'mean_lv')
    del mean_lv['last_visit']
    return mean_lv
