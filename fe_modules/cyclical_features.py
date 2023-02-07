import numpy as np
import pandas as pd


def generate_cyclical_features(df: pd.DataFrame, col_name: str, period: int, start_num: int = 0) -> pd.DataFrame:
    kwargs = {
        f'sin_{col_name}': lambda x: np.sin(2 * np.pi * (x[col_name] - start_num) / period),
        f'cos_{col_name}': lambda x: np.cos(2 * np.pi * (x[col_name] - start_num) / period)
    }
    return df.assign(**kwargs)
