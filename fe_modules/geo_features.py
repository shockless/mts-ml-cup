from aggregates import get_agg_mean, get_agg_min, get_agg_max
import pandas as pd
from modules.memory_utils import pandas_reduce_mem_usage
from geopy.distance import geodesic
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.spatial.distance import cdist
import numpy as np

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


def map_grid(df: pd.DataFrame, col=30, row=90):
    nodes = np.array([
        [68.970513, 33.074320],
        [68.970513, 177.518731],
        [44.616489, 33.074320],
        [44.616489, 177.518731]
    ])
    map_grider = MapGridTransformer(nodes, col, row)
    map_grider.fit()
    df['grid'] = map_grider.transform(df[["latitude", "longitude"]])
    return df


class MapGridTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, loc, col, row, col_names=COL_NAMES):
        self.location_mh = loc
        self.col = col
        self.row = row
        self.col_names = col_names
        self.lat = col_names["lat"]
        self.lon = col_names["lon"]
        self.lat_min, self.lat_max = loc[:, 0].min(), loc[:, 0].max()
        self.lon_min, self.lon_max = loc[:, 1].min(), loc[:, 1].max()

    def fit(self):
        self.walls = [(self.location_mh[:, 0].max() - self.location_mh[:, 0].min()) / self.col, \
                      (self.location_mh[:, 1].max() - self.location_mh[:, 1].min()) / self.row]

        self.circles_loc = np.array([(((self.location_mh[:, 0].min() + i * (self.walls[0] / 2))), \
                                      (self.location_mh[:, 1].min() + j * (self.walls[1] / 2))) \
                                     for i in range(1, (self.col) * 2, 2) for j in range(1, (self.row) * 2, 2)])

        return self.walls, self.circles_loc

    def transform(self, X):
        idx = cdist(X.loc[:, [self.lat, self.lon]],
                    self.circles_loc).argmin(axis=1)
        qry = f"@self.lat_min <= {self.lat} <= @self.lat_max \
               and @self.lon_min <= {self.lon} <= @self.lon_max"
        valid = X.eval(qry).to_numpy()
        idx[~valid] = 0
        return idx
