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

    def fit(self, df: pd.DataFrame, rows: str = "user_id", columns: str = "url_host", target: str = 'request_cnt',
            agg_fn: str = "sum"):
        data_agg = df.groupby([rows, columns])[[rows, columns, target]].agg(
            {target: agg_fn}).reset_index().rename(columns={target: target + '_' + agg_fn})
        url_set = set(df[columns])
        self.url_dict = {url: idurl for url, idurl in zip(url_set, range(len(url_set)))}
        usr_set = set(df[rows])
        self.usr_dict = {usr: user_id for usr, user_id in zip(usr_set, range(len(usr_set)))}
        values = np.array(data_agg[target + '_' + agg_fn])
        row = np.array(data_agg[rows].map(self.usr_dict))
        cols = np.array(data_agg[columns].map(self.url_dict))
        mat = scipy.sparse.coo_matrix((values, (row, cols)), shape=(row.max() + 1, cols.max() + 1))
        self.als.fit(mat)

    def get_embeddings(self, emb_name: str = "emb"):
        u_factors = self.als.user_factors
        inv_usr_map = {v: k for k, v in self.usr_dict.items()}
        usr_emb = pd.DataFrame(u_factors)
        usr_emb = usr_emb.rename(columns={column: f"{emb_name}_" + str(column) for column in usr_emb.columns})
        usr_emb['user_id'] = usr_emb.index.map(inv_usr_map)
        return usr_emb

    def save_model(self, path_to_model: str = "als"):
        joblib.dump(self, f"{path_to_model}.joblib")
