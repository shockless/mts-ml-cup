import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold

RANDOM_STATE = 42
DEFAULT_PARAMS = {
    "max_depth": 3,
    "iterations": 500,
    "random_state": RANDOM_STATE
}


def cb_pipeline(df: pd.DataFrame, target_feature: str, new_feature: str = None,
                cb_params: dict = DEFAULT_PARAMS, n_folds=5) -> pd.DataFrame:
    y = df[target_feature]
    X = df.drop(columns=[target_feature])
    preds = y.copy()
    kf = KFold(n_splits=n_folds, random_state=RANDOM_STATE, shuffle=True)
    i = 0
    for train_index, val_index in kf.split(X):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        model = CatBoostRegressor(**cb_params)
        model.fit(X_train, y_train,
                  eval_set=(X_val, y_val), use_best_model=True)

        preds[val_index] = model.predict(X_val)
        model.save_model(fname='cb_models/feat_' + target_feature + f'_{i}_fold',
                         format='cbm')
        i += 1
    if new_feature is None:
        feat_name = target_feature + '_preds'
    else:
        feat_name = new_feature
    df[feat_name] = abs(df[target_feature] - preds)
    return df
