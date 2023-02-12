from aggregates import get_agg_mean, get_agg_min, get_agg_max
import pandas as pd
from modules.memory_utils import pandas_reduce_mem_usage
from geopy.distance import geodesic

LARGE_CITIES = {'Moscow': (55.755864, 37.617698),
                'SaintP': (59.938955, 30.315644),
                'Novosibirsk': (55.030204, 82.920430),
                'Ekaterinburg': (56.838011, 60.597474),
                'Vladivostok': (43.115542, 131.885494),
                }

COL_NAMES = dict(
    lat="latitude",
    lon="longitude",
)


def mean_first_visit(df: pd.DataFrame) -> pd.DataFrame:
    first_visit = get_agg_min(df, agg_col=['user_id', 'date'], col='first_visit', target_col='part_of_day')
    mean_fv = get_agg_mean(first_visit, agg_col=['user_id', 'date'], col='mean_fv', target_col='first_visit')
    del mean_fv['first_visit']
    return mean_fv


def mean_last_visit(df: pd.DataFrame) -> pd.DataFrame:
    last_visit = get_agg_max(df, agg_col=['user_id', 'date'], col='last_visit', target_col='part_of_day')
    mean_lv = get_agg_mean(last_visit, agg_col=['user_id', 'date'], col='mean_fv', target_col='last_visit')
    del mean_lv['last_visit']
    return mean_lv


def map_cities(df, cities_path='cities_finally.csv'):
    cities = pandas_reduce_mem_usage(
        pd.read_csv(cities_path))
    df = df.merge(cities, on="city_name", how="left")
    return df


def geo_dist(self, loc1: tuple, loc2: tuple) -> float:
    try:
        dist = geodesic(loc1, loc2).km
    except ValueError:
        dist = -1
    return dist


def dist_to_large_cities(df) -> pd.DataFrame:
    df['lat_long'] = tuple(zip(df.latitude, df.longitude))

    for name, loc in LARGE_CITIES.items():
        col_name = f'dist_to_{name}'
        df[col_name] = df['lat_long'].apply(lambda x: geodesic(x, loc).km)
    del df['lat_long']
    return df
