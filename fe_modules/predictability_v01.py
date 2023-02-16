import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import StratifiedGroupKFold

RANDOM_STATE = 42
DEFAULT_PARAMS = {
    "max_depth": 3,
    "iterations": 500,
    "random_state": RANDOM_STATE
}


def cb_predictability(df: pd.DataFrame, target_feature: str, new_feature: str = None,
                      cb_params: dict = DEFAULT_PARAMS, n_folds=5) -> pd.DataFrame:
    y = df[target_feature]
    users = df['user_id']
    X = df.drop(columns=[target_feature, 'user_id'])
    preds = np.zeros(shape=y.shape)
    sgkf = StratifiedGroupKFold(n_splits=n_folds, random_state=RANDOM_STATE, shuffle=True)
    for i, (train_index, val_index) in enumerate(sgkf.split(X, y, groups=users)):
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
        feat_name = target_feature + '_predictability'
    else:
        feat_name = new_feature
    df[feat_name] = abs(y - preds)
    return df
