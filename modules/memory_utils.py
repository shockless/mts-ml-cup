import numpy as np
import polars as pl
import pandas as pd
from IPython import get_ipython


def pandas_reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
    """
    iterate through all the columns of a dataframe and modify the data type
    to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print("Memory usage of dataframe is {:.2f} MB".format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype("category")

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))

    return df


def polars_reduce_mem_usage(df: pl.DataFrame) -> pl.DataFrame:
    """
    iterate through all the columns of a dataframe and modify the data type
    to reduce memory usage.
    """
    start_mem = df.estimated_size() / 1024 ** 2
    print("Memory usage of dataframe is {:.2f} MB".format(start_mem))
    print("Memory usage of dataframe is {:.2f} GB".format(start_mem  / 1024 ))


    for col in df.columns:
        col_type = df[col].dtype
        print(str(col_type)[:3].lower())

        if str(col_type)[:3].lower() in {"int", "flo"}:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3].lower() == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df = df.with_column(pl.col(col).cast(pl.Int8, strict=True))
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df = df.with_column(pl.col(col).cast(pl.Int16, strict=True))
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df = df.with_column(pl.col(col).cast(pl.Int32, strict=True))
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df = df.with_column(pl.col(col).cast(pl.Int64, strict=True))
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df = df.with_column(pl.col(col).cast(pl.Float32, strict=True))
                else:
                    df = df.with_column(pl.col(col).cast(pl.Float64, strict=True))

    end_mem = df.estimated_size()  / 1024 ** 2
    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    print("Memory usage  after optimization is {:.2f} GB".format(end_mem  / 1024))
    print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))

    return df


def my_reset(*varnames):
    """
    varnames are what you want to keep
    """
    globals_ = globals()
    to_save = {v: globals_[v] for v in varnames}
    to_save["my_reset"] = my_reset  # let's keep this function by default
    del globals_
    get_ipython().magic("reset")
    globals().update(to_save)
