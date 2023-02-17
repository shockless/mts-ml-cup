import implicit
import pandas as pd
import numpy as np
import scipy
import joblib


class ALSWrapper:
    def __init__(self, factors=50, iterations=30, use_gpu=False, regularization=0.1):
        self.als = implicit.als.AlternatingLeastSquares(factors=factors,
                                                        iterations=iterations, use_gpu=use_gpu,
                                                        regularization=regularization)
        self.usr_dict = None
        self.url_dict = None

    def fit(self, df: pd.DataFrame, target: str = 'request_cnt'):
        data_agg = df.groupby(['user_id', 'url_host'])[['user_id', 'url_host', target]].agg(
            {target: 'sum'}).reset_index().rename(columns={target: target + '_sum'})
        url_set = set(df['url_host'])
        self.url_dict = {url: idurl for url, idurl in zip(url_set, range(len(url_set)))}
        usr_set = set(df['user_id'])
        self.usr_dict = {usr: user_id for usr, user_id in zip(usr_set, range(len(usr_set)))}
        values = np.array(data_agg[target + '_sum'])
        rows = np.array(data_agg['user_id'].map(self.usr_dict))
        cols = np.array(data_agg['url_host'].map(self.url_dict))
        mat = scipy.sparse.coo_matrix((values, (rows, cols)), shape=(rows.max() + 1, cols.max() + 1))
        self.als.fit(mat)

    def get_embeddings(self):
        u_factors = self.als.user_factors
        inv_usr_map = {v: k for k, v in self.usr_dict.items()}
        usr_emb = pd.DataFrame(u_factors)
        usr_emb['user_id'] = usr_emb.index.map(inv_usr_map)
        return usr_emb

    def save_model(self, path_to_model: str = "als"):
        joblib.dump(self, f"{path_to_model}.joblib")
