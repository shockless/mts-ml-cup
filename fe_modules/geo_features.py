from aggregates import get_agg_mean, get_agg_min, get_agg_max
import pandas as pd


def mean_first_visit(df: pd.DataFrame) -> pd.DataFrame:
    first_visit = get_agg_min(df, agg_col=['user_id', 'date'], col='first_visit', target_col='part_of_day')
    mean_fv = get_agg_mean(first_visit, agg_col=['user_id', 'date'], col='mean_fv', target_col='first_visit')
    del mean_fv['first_visit']
    return mean_fv


def mean_last_visit(df: pd.DataFrame) -> pd.DataFrame:
    last_visit = get_agg_max(df, agg_col=['user_id', 'date'], col='last_visit', target_col='part_of_day')
    mean_fv = get_agg_mean(last_visit, agg_col=['user_id', 'date'], col='mean_fv', target_col='last_visit')
    del mean_fv['last_visit']
    return mean_fv
