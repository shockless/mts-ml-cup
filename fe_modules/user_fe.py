from typing import Union

import pandas as pd
import numpy as np

from modules.memory_utils import pandas_reduce_mem_usage

from fe_modules.preprocessing import clean_os_type


class UserFE:
    def __init__(self, df: pd.DataFrame):
        self.df = df.drop_duplicates("user_id").reset_index()
        self.df = self.df.drop(["index", "region_name", "city_name", "url_host", "price", "request_cnt"], axis=1)
        self.df = clean_os_type(self.df)
        self.pandas_reduce_mem_usage()

    def get_agg_mean(self,
                     df: pd.DataFrame,
                     agg_col: Union[str, list] = "user_id",
                     target_col: str = None,
                     alias: str = None):
        if alias:
            col_name = alias
        else:
            col_name = f'{agg_col}_group_mean'

        if isinstance(agg_col, str):
            agg_col = [agg_col]

        agg = df.groupby(agg_col)[target_col].mean().to_frame().rename(columns={target_col: col_name}).reset_index()
        self.df = self.df.merge(agg, how="left", on=agg_col)

    def get_agg_count(self,
                      df: pd.DataFrame,
                      agg_col: Union[str, list] = "user_id",
                      target_col: str = None,
                      alias: str = None):
        if alias:
            col_name = alias
        else:
            col_name = f'{agg_col}_group_count'

        if isinstance(agg_col, str):
            agg_col = [agg_col]

        agg = df.groupby(agg_col)[target_col].count().to_frame().rename(columns={target_col: col_name}).reset_index()
        self.df = self.df.merge(agg, how="left", on=agg_col)

    def get_agg_sum(self,
                    df: pd.DataFrame,
                    agg_col: Union[str, list] = "user_id",
                    target_col: str = None,
                    alias: str = None):
        if alias:
            col_name = alias
        else:
            col_name = f'{agg_col}_group_sum'

        if isinstance(agg_col, str):
            agg_col = [agg_col]

        agg = df.groupby(agg_col)[target_col].sum().to_frame().rename(columns={target_col: col_name}).reset_index()
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
            col_name = f'{agg_col}_mode'

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

    def pandas_reduce_mem_usage(self, *args):
        self.df = pandas_reduce_mem_usage(self.df, args)