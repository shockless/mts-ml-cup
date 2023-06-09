import numpy as np
import pandas as pd
import polars as pl
from tqdm.auto import tqdm
from IPython import get_ipython


def pandas_reduce_mem_usage(df: pd.DataFrame, columns=None) -> pd.DataFrame:
    """
    iterate through all the columns of a dataframe and modify the external_data type
    to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print("Memory usage of dataframe is {:.2f} MB".format(start_mem))

    columns = tqdm(columns) if columns else tqdm(df.columns)

    for col in columns:
        col_type = df[col].dtype
        if str(col_type)[:3] == "int":
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
            elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                df[col] = df[col].astype(np.int64)
        elif str(col_type)[:5] == "float":
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
            else:
                df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))

    return df


def polars_reduce_mem_usage(df: pl.DataFrame) -> pl.DataFrame:
    """
    iterate through all the columns of a dataframe and modify the external_data type
    to reduce memory usage.
    """
    start_mem = df.estimated_size() / 1024 ** 2
    print("Memory usage of dataframe is {:.2f} MB".format(start_mem))
    print("Memory usage of dataframe is {:.2f} GB".format(start_mem / 1024))

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

    end_mem = df.estimated_size() / 1024 ** 2
    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    print("Memory usage  after optimization is {:.2f} GB".format(end_mem / 1024))
    print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))

    return df

    
def pandas_string_to_cat(df: pl.DataFrame, columns: list):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print("Memory usage of dataframe is {:.2f} MB".format(start_mem))
    print("Memory usage of dataframe is {:.2f} GB".format(start_mem / 1024))

    for col in tqdm(columns):
        df[col] = pd.Categorical(df[col]).codes.astype(np.uint64)

        c_min = df[col].min()
        c_max = df[col].max()
        
        if c_max < np.iinfo(np.uint8).max:
            df[col] = df[col].astype(np.uint8)
        elif c_max < np.iinfo(np.uint16).max:
            df[col] = df[col].astype(np.uint16)
        elif c_max < np.iinfo(np.uint32).max:
            df[col] = df[col].astype(np.uint32)
        elif c_max < np.iinfo(np.uint64).max:
            df[col] = df[col].astype(np.uint64)

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))

    return df


def polars_string_to_cat(df: pl.DataFrame, columns: list):
    start_mem = df.estimated_size() / 1024 ** 2
    print("Memory usage of dataframe is {:.2f} MB".format(start_mem))
    print("Memory usage of dataframe is {:.2f} GB".format(start_mem / 1024))

    for col in columns:
        df = df.with_column(pl.col(col).cast(pl.Categorical, strict=True).cast(pl.Int64, strict=True))

        c_min = df[col].min()
        c_max = df[col].max()
        
        print(type(c_min), type(c_max), c_min, c_max)

        if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
            df = df.with_column(pl.col(col).cast(pl.Int8, strict=True))
        elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
            df = df.with_column(pl.col(col).cast(pl.Int16, strict=True))
        elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
            df = df.with_column(pl.col(col).cast(pl.Int32, strict=True))
        elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
            df = df.with_column(pl.col(col).cast(pl.Int64, strict=True))

    end_mem = df.estimated_size() / 1024 ** 2
    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    print("Memory usage  after optimization is {:.2f} GB".format(end_mem / 1024))
    print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))

    return df


def get_suitable_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    for col in tqdm(df.columns):
        col_type = df[col].dtype

        if col_type not in [object, np.uint8, np.uint16, np.uint32, np.uint64]:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "flo":
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    return df

