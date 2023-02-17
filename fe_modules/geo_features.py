import pandas as pd
from modules.memory_utils import pandas_reduce_mem_usage
from geopy.distance import geodesic
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.spatial.distance import cdist
import numpy as np
from tqdm.auto import tqdm

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


def get_travel(city_name_col):
    return (city_name_col != city_name_col.shift(1)).sum() - 1


def process_utc(cities: pd.DataFrame, timezone_col: str = "timezone"):
    cities[timezone_col] = cities[timezone_col].apply(lambda x: int(x.split("+")[1]))
    return cities


def map_cities(df: pd.DataFrame, folder_path: str = "../external_data", cities_path: str = 'cities_finally.csv'):
    cities = pandas_reduce_mem_usage(
        process_utc(
            pd.read_csv(f"{folder_path}/{cities_path}")
        )
    )
    df = df.merge(cities, on="city_name", how="left")
    return df


def geo_dist(loc1: tuple, loc2: tuple) -> float:
    try:
        dist = geodesic(loc1, loc2).km
    except ValueError:
        dist = -1
    return dist


def dist_to_large_cities(df) -> pd.DataFrame:
    unique_cities = df.drop_duplicates(subset=["city_name"])[["city_name", "geo_lat", "geo_lon"]]
    unique_cities['lat_long'] = tuple(zip(unique_cities.geo_lat, unique_cities.geo_lon))

    for name, loc in tqdm(LARGE_CITIES.items()):
        col_name = f'dist_to_{name}'
        unique_cities[col_name] = unique_cities['lat_long'].apply(lambda x: geodesic(x, loc).km)

    del unique_cities['lat_long']

    unique_cities = pandas_reduce_mem_usage(unique_cities)
    df = df.merge(unique_cities.drop(labels=["geo_lat", "geo_lon"], axis=1), how="left", on="city_name")

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


def get_agg_amount_of_travel(df: pd.DataFrame,
                             agg_col: str = "user_id",
                             target_col: str = 'city_name',
                             timestamp_col: str = 'timestamp',
                             alias: str = None,
                             sort: bool = False) -> pd.DataFrame:
    if alias:
        col_name = alias
    else:
        col_name = f'{agg_col}_amount_of_travel'

    df = df.merge(df.sort_values(timestamp_col).groupby(agg_col)[target_col].agg(
        amount_of_travel=get_travel
    ).rename(columns={'amount_of_travel': col_name})
                  , how='left', on=agg_col)
    if sort:
        return df.sort_values(by=agg_col)

    return df


def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km


def get_dist(points):
    points = np.stack(points.to_numpy())
    point_shifted = np.roll(points, 1, axis=0)
    point_shifted[0] = np.array([np.nan, np.nan, np.nan])
    temp = (point_shifted[:, 2] != points[:, 2])
    points = np.delete(points[temp], 2, 1).astype(np.float32)[1:]
    point_shifted = np.delete(point_shifted[temp], 2, 1).astype(np.float32)[1:]
    return haversine_np(points[:, 0],
                        points[:, 1],
                        point_shifted[:, 0],
                        point_shifted[:, 1]).mean()


def get_agg_distance_of_travel(df: pd.DataFrame,
                               agg_col: str = "user_id",
                               target_col_lat: str = 'geo_lat',
                               target_col_lon: str = 'geo_lon',
                               timestamp_col: str = 'timestamp',
                               city_name_col: str = 'city_name',
                               alias: str = None,
                               sort: bool = False) -> pd.DataFrame:
    if alias:
        col_name = alias
    else:
        col_name = f'{agg_col}_mean_travel_distance'
    df['lat_lon'] = df[[target_col_lat, target_col_lon, city_name_col]].progress_apply(lambda r: tuple(r),
                                                                                       axis=1).apply(np.array)

    df = df.merge(df.sort_values(timestamp_col).groupby(agg_col)['lat_lon']
                  .agg(geometry=get_dist).fillna(0).astype(np.float32)
                  .rename(columns={'geometry': col_name}), how='left', on=agg_col)

    del df['lat_lon']

    if sort:
        return df.sort_values(by=agg_col)

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
        self.walls = [(self.location_mh[:, 0].max() - self.location_mh[:, 0].min()) / self.col,
                      (self.location_mh[:, 1].max() - self.location_mh[:, 1].min()) / self.row]

        self.circles_loc = np.array([(((self.location_mh[:, 0].min() + i * (self.walls[0] / 2))),
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
