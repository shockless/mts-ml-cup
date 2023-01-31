import polars as pl


def add_cat_features(df: pl.DataFrame, feature_a: str, feature_b: str, alias: str = None) -> pl.Series:
    new_feature = df[feature_a] + "_" + df[feature_b]

    if alias:
        new_feature = new_feature.alias(alias)
    else:
        new_feature = new_feature.alias(f"{feature_a}_{feature_b}")

    return new_feature

