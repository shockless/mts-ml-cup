import pandas as pd
import category_encoders as ce
import joblib


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
