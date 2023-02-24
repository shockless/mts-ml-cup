from typing import Union

import pandas as pd
import numpy as np

from modules.memory_utils import pandas_reduce_mem_usage

from fe_modules.preprocessing import clean_os_type
from fe_modules.geo_features import get_travel, get_dist


class UserFE:
    def __init__(self):
        self.df = None

    def load_df(self, df: pd.DataFrame):
        self.df = df.drop_duplicates("user_id").reset_index()
        self.df = self.df.drop(["index", "region_name", "city_name", "url_host", "price", "request_cnt",
                                "date", "part_of_day"],
                               axis=1)
        self.df = clean_os_type(self.df)
        self.pandas_reduce_mem_usage()

    def get_agg(self,
                df: pd.DataFrame,
                agg_col: Union[str, list] = "user_id",
                target_col: str = None,
                agg_name: str = None,
                alias: str = None):

        if alias:
            col_name = alias
        else:
            col_name = f'{agg_col}_group_{agg_name}'

        if isinstance(agg_col, str):
            agg_col = [agg_col]

        if agg_name == "mean":
            agg = df.groupby(agg_col)[target_col].mean().to_frame().rename(columns={target_col: col_name}).reset_index()
        elif agg_name == "count":
            agg = df.groupby(agg_col)[target_col].count().to_frame().rename(
                columns={target_col: col_name}).reset_index()
        elif agg_name == "sum":
            agg = df.groupby(agg_col)[target_col].sum().to_frame().rename(columns={target_col: col_name}).reset_index()
        elif agg_name == "max":
            agg = df.groupby(agg_col)[target_col].max().to_frame().rename(columns={target_col: col_name}).reset_index()
        elif agg_name == "min":
            agg = df.groupby(agg_col)[target_col].min().to_frame().rename(columns={target_col: col_name}).reset_index()
        elif agg_name == "nunique":
            agg = df.groupby(agg_col)[target_col].nunique().to_frame().rename(
                columns={target_col: col_name}).reset_index()
        else:
            raise Exception(f"'agg_name' can't take value '{agg_name}'")

        self.df = self.df.merge(agg, how="left", on=agg_col)

    def get_top_n_mode(self,
                       df: pd.DataFrame,
                       agg_col: Union[str, list] = "user_id",
                       target_col: str = None,
                       n: int = 1,
                       alias: str = None):
        if alias:
            col_name = alias
        else:
            col_name = f'{target_col}_mode'

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

        self.df = self.df.merge(agg, how="left", on=agg_col)

    def get_ratio_part_of_day(self,
                              df: pd.DataFrame,
                              agg_col: Union[str, list] = "user_id"):
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

        self.df = self.df.merge(agg, how="left", on=agg_col)

        self.df["morning"] = self.df["morning"].astype(np.float64)
        self.df["day"] = self.df["day"].astype(np.float64)
        self.df["evening"] = self.df["evening"].astype(np.float64)
        self.df["night"] = self.df["night"].astype(np.float64)

    def get_timespan(self,
                     df: pd.DataFrame,
                     agg_col: Union[str, list] = "user_id",
                     date_col: str = "datetime",
                     alias: str = None,
                     scaler: int = 1e11):

        if alias is None:
            alias = "timespan"

        if isinstance(agg_col, str):
            agg_col = [agg_col]

        self.get_agg(df, agg_col=agg_col, target_col=date_col, agg_name="min", alias="date_min")
        self.get_agg(df, agg_col=agg_col, target_col=date_col, agg_name="max", alias="date_max")

        self.df[alias] = (pd.DatetimeIndex(self.df["date_max"]).astype(int) - pd.DatetimeIndex(
            self.df["date_min"]).astype(int)) / scaler
        self.df = self.df.drop(["date_max", "date_min"], axis=1)

    def get_ratio_request_timespan(self, alias: str = None):
        if alias is None:
            alias = "ratio_request_timespan"
        self.df[alias] = self.df["request_sum"] / self.df["timespan"]
        self.df.loc[self.df["timespan"] == 0, alias] = 0

    def get_ratio(self,
                  df: pd.DataFrame,
                  agg_col: Union[str, list] = "user_id",
                  ratio_col: str = "url_host",
                  n: int = 3):
        if isinstance(agg_col, str):
            agg_col = [agg_col]

        def ratio(s):
            dtemp = s.value_counts()

            temp = dtemp[:n].index
            temp = np.append(temp, np.array(["<blank>"] * (n - len(temp))))

            dtemp["<blank>"] = 0

            aggregates = np.zeros(n)
            for i in range(len(temp)):
                aggregates[i] = dtemp[temp[i]] / s.shape[0]

            return aggregates

        agg = df.groupby(agg_col)[ratio_col].agg(ratio).to_frame()
        agg = pd.DataFrame(np.concatenate((np.expand_dims(agg.index.to_numpy().astype(object), axis=1),
                                           np.stack(agg[ratio_col].values)), axis=1),
                           columns=agg_col + [f"{ratio_col}_ratio_{i}" for i in range(n)])

        self.df = self.df.merge(agg, how="left", on=agg_col)

    def get_first_visit_sec(self,
                            df: pd.DataFrame,
                            agg_col: Union[str, list] = "user_id",
                            date_col: str = "datetime",
                            alias: str = None,
                            scaler: int = 1e11):

        if alias is None:
            alias = "first_visit_sec"

        if isinstance(agg_col, str):
            agg_col = [agg_col]

        self.get_agg(df, agg_col=agg_col, target_col=date_col, agg_name="min", alias="date_min")

        self.df[alias] = pd.DatetimeIndex(self.df["date_min"]).astype(int) / scaler

        self.df = self.df.drop(["date_min"], axis=1)

    def get_agg_amount_of_travel(self,
                                 df: pd.DataFrame,
                                 agg_col: str = "user_id",
                                 target_col: str = "city_name",
                                 timestamp_col: str = "timestamp",
                                 alias: str = None):
        if alias:
            col_name = alias
        else:
            col_name = f'{agg_col}_amount_of_travel'

        self.df = self.df.merge(df.sort_values(timestamp_col).groupby(agg_col)[target_col].agg(
            amount_of_travel=get_travel
        ).rename(columns={'amount_of_travel': col_name})
                                , how='left', on=agg_col)

    def get_agg_distance_of_travel(self,
                                   df: pd.DataFrame,
                                   agg_col: str = "user_id",
                                   target_col_lat: str = "geo_lat",
                                   target_col_lon: str = "geo_lon",
                                   timestamp_col: str = "timestamp",
                                   city_name_col: str = "city_name",
                                   alias: str = None):
        if alias:
            col_name = alias
        else:
            col_name = f'{agg_col}_mean_travel_distance'

        self.df = self.df.merge(df.sort_values(timestamp_col).groupby(agg_col)[[target_col_lat,
                                                                                target_col_lon,
                                                                                city_name_col]].progress_apply(
            lambda x: get_dist(x)).
                                fillna(0).astype(np.float32).to_frame(col_name), how='left', on=agg_col)

    def pandas_reduce_mem_usage(self, *args):
        self.df = pandas_reduce_mem_usage(self.df, args)

    def save(self, path: str):
        self.df.to_parquet(path, compression='gzip')

    def load(self, path: str):
        self.df = pandas_reduce_mem_usage(pd.read_parquet(path))
