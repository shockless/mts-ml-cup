import category_encoders as ce
import joblib
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold

from modules.memory_utils import pandas_reduce_mem_usage


class TargetEncoderWrapper:
    def __init__(self):
        self.tenc = ce.TargetEncoder()

    def fit(self, df: pd.DataFrame, feature_col: str, target_col: str):
        self.tenc.fit(df[feature_col], df[target_col])

    def encode(self, df: pd.DataFrame, feature_col: str) -> pd.DataFrame:
        df[f"te_{feature_col}"] = self.tenc.transform(df[feature_col])
        return df

    def save(self, path: str):
        with open(path, "wb") as fout:
            joblib.dump(self.tenc, file=fout)

    def load(self, path: str):
        self.tenc = joblib.load(path)


class CatBoostEncoderWrapper:
    def __init__(self,
                 cat_features,
                 sort_col: str = "timestamp",
                 verbose: int = 1,
                 drop_invariant: bool = False,
                 return_df: bool = True,
                 n_folds: int = 5,
                 random_state: int = 42,
                 ):
        self.verbose = verbose
        self.cat_features = cat_features
        self.sort_col = sort_col
        self.drop_invariant = drop_invariant
        self.return_df = return_df
        self.n_folds = n_folds
        self.random_state = random_state

        self.oof_encoders = [
            ce.cat_boost.CatBoostEncoder(
                verbose=self.verbose,
                cols=self.cat_features,
                drop_invariant=self.drop_invariant,
                return_df=self.return_df
            ) for i in range(self.n_folds)
        ]

    def fit_transform(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        oof_parts = []
        oof_target = []

        skf = KFold(n_splits=self.n_folds, random_state=self.random_state, shuffle=True)
        for i, (train_index, val_index) in enumerate(skf.split(df, df[target_col])):
            train_df = df.iloc[train_index].sort_values(self.sort_col)
            val_df = df.iloc[val_index].sort_values(self.sort_col)

            cbe = self.oof_encoders[i]
            cbe.fit(X=train_df.drop(columns=[target_col]), y=train_df[target_col])
            oof_part = cbe.transform(val_df.drop(columns=[target_col]))

            oof_parts.append(oof_part)
            oof_target.append(val_df[target_col])

        target = pd.concat(oof_target)
        final_df = pd.concat(oof_parts)
        final_df[target_col] = target

        return final_df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        oof_parts = []

        for encoder in self.oof_encoders:
            transformer_cat_features = pandas_reduce_mem_usage(encoder.transform(df))[self.cat_features].to_numpy()
            oof_parts.append(transformer_cat_features)

        oof_parts = sum(oof_parts) / self.n_folds
        df[self.cat_features] = oof_parts

        return df

    def save(self, folder_path:str, model_name: str):
        joblib.dump(self, f"{folder_path}/{model_name}.joblib")
