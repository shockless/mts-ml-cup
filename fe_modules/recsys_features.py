import implicit
import pandas as pd
import numpy as np
import scipy
import joblib

from scipy import sparse
import os
import bottleneck as bn
from copy import deepcopy
import sys
import random
import torch
from torch import nn
from torch.nn import functional as F
from tqdm.notebook import tqdm


class ALSWrapper:
    def __init__(self, factors=50, iterations=30, use_gpu=False, regularization=0.1, alpha=1.0, random_state=42):
        self.als = implicit.als.AlternatingLeastSquares(factors=factors,
                                                        iterations=iterations,
                                                        use_gpu=use_gpu,
                                                        alpha=alpha,
                                                        regularization=regularization,
                                                        random_state=random_state)
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


class RecVAEWrapper:
    def __init__(self,
                 hidden_dim: int = 600,
                 latent_dim: int = 200
                 ):
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.url_dict = None
        self.usr_dict = None
        self.user_embs = None

    def fit(self, df: pd.DataFrame, rows: str = "user_id", columns: str = "url_host", target: str = 'request_cnt',
            agg_fn: str = "sum", threshold: float = None, batch_size: int = 500, n_epochs: int = 20, beta=0.2,
            gamma=0.005):
        data_agg = df.groupby([rows, columns])[[rows, columns, target]].agg(
            {target: agg_fn}).reset_index().rename(columns={target: target + '_' + agg_fn})
        if threshold is None:
            threshold = data_agg[target + '_' + agg_fn].mean()
        data_agg[target + '_' + agg_fn] = (data_agg[target + '_' + agg_fn] > threshold).astype(int)
        url_set = set(df[columns])
        self.url_dict = {url: idurl for url, idurl in zip(url_set, range(len(url_set)))}
        usr_set = set(df[rows])
        self.usr_dict = {usr: user_id for usr, user_id in zip(usr_set, range(len(usr_set)))}
        n_items = len(url_set)
        n_users = len(usr_set)
        data_agg[rows] = data_agg[rows].map(self.usr_dict)
        data_agg[columns] = data_agg[columns].map(self.url_dict)
        mat = sparse.csr_matrix((np.ones_like(data_agg[rows]),
                                 (data_agg[rows], data_agg[columns])), dtype='float64',
                                shape=(n_users, n_items))
        batches_per_epoch = int(np.ceil(float(n_users) / batch_size))
        anneal_cap = 0.2
        model_kwargs = {
            'hidden_dim': self.hidden_dim,
            'latent_dim': self.latent_dim,
            'input_dim': n_items
        }
        self.model = VAE(**model_kwargs).to(self.device)
        learning_kwargs = {
            'train_data': mat,
            'batch_size': batch_size,
            'beta': beta,
            'gamma': gamma
        }
        decoder_params = set(self.model.decoder.parameters())
        encoder_params = set(self.model.encoder.parameters())
        optimizer_encoder = torch.optim.Adam(encoder_params, lr=5e-4)
        optimizer_decoder = torch.optim.Adam(decoder_params, lr=5e-4)
        not_alternating = True

        print('Training...')
        for epoch in tqdm(range(n_epochs)):
            if not_alternating:
                self.run(opts=[optimizer_encoder, optimizer_decoder], n_epochs=1, dropout_rate=0.5, **learning_kwargs)
            else:
                self.run(opts=[optimizer_encoder], n_epochs=5, dropout_rate=0.5, **learning_kwargs)
                self.model.update_prior()
                self.run(opts=[optimizer_decoder], n_epochs=5, dropout_rate=0, **learning_kwargs)

        print('Generate user embeddings...')
        for batch in tqdm(self.generate(batch_size=batch_size, data_in=mat, shuffle=True)):
            with torch.no_grad():
                self.model.eval()
                ratings = batch.get_ratings_to_dev()
                pred = self.model(ratings, calculate_loss=False)
                mu, _ = self.model.encoder(ratings, dropout_rate=0)
                if self.user_embs is None:
                    self.user_embs = mu
                else:
                    self.user_embs = torch.cat([self.user_embs, mu], dim=0)
        self.user_embs = self.user_embs.cpu().detach().numpy()

    def get_embeddings(self, emb_name: str = 'emb'):
        u_factors = self.user_embs
        inv_usr_map = {v: k for k, v in self.usr_dict.items()}
        usr_emb = pd.DataFrame(u_factors)
        usr_emb = usr_emb.rename(columns={column: f"{emb_name}_" + str(column) for column in usr_emb.columns})
        usr_emb['user_id'] = usr_emb.index.map(inv_usr_map)
        return usr_emb

    def save_model(self, path_to_model: str = "vae"):
        joblib.dump(self, f"{path_to_model}.joblib")

    def run(self, opts, train_data, batch_size, n_epochs, beta, gamma, dropout_rate):
        self.model.train()
        for epoch in range(n_epochs):
            for batch in self.generate(batch_size=batch_size, data_in=train_data, shuffle=True):
                ratings = batch.get_ratings_to_dev()

                (ml, kl), loss = self.model(ratings, beta=beta, gamma=gamma, dropout_rate=dropout_rate)
                if torch.isnan(ml):
                    continue
                loss.backward()

                for optimizer in opts:
                    optimizer.step()

                for optimizer in opts:
                    optimizer.zero_grad()

    def generate(self, batch_size, data_in, data_out=None, shuffle=False, samples_perc_per_epoch=1):
        assert 0 < samples_perc_per_epoch <= 1

        total_samples = data_in.shape[0]
        samples_per_epoch = int(total_samples * samples_perc_per_epoch)

        if shuffle:
            idxlist = np.arange(total_samples)
            np.random.shuffle(idxlist)
            idxlist = idxlist[:samples_per_epoch]
        else:
            idxlist = np.arange(samples_per_epoch)

        for st_idx in range(0, samples_per_epoch, batch_size):
            end_idx = min(st_idx + batch_size, samples_per_epoch)
            idx = idxlist[st_idx:end_idx]
            yield Batch(self.device, idx, data_in, data_out)

    def evaluate(self, data_in, data_out, metrics, samples_perc_per_epoch=1, batch_size=500):
        metrics = deepcopy(metrics)
        self.model.eval()

        for m in metrics:
            m['score'] = []

        for batch in self.generate(batch_size=batch_size,
                                   data_in=data_in,
                                   data_out=data_out,
                                   samples_perc_per_epoch=samples_perc_per_epoch
                                   ):

            ratings_in = batch.get_ratings_to_dev()
            ratings_out = batch.get_ratings(is_out=True)

            ratings_pred = self.model(ratings_in, calculate_loss=False).cpu().detach().numpy()
            print(ratings_pred[0])

            if not (data_in is data_out):
                ratings_pred[batch.get_ratings().nonzero()] = -np.inf

            for m in metrics:
                m['score'].append(m['metric'](ratings_pred, ratings_out, k=m['k']))


def swish(x):
    return x.mul(torch.sigmoid(x))


def log_norm_pdf(x, mu, logvar):
    return -0.5 * (logvar + np.log(2 * np.pi) + (x - mu).pow(2) / logvar.exp())


class CompositePrior(nn.Module):
    def __init__(self, hidden_dim, latent_dim, input_dim, mixture_weights=[3 / 20, 3 / 4, 1 / 10]):
        super(CompositePrior, self).__init__()

        self.mixture_weights = mixture_weights

        self.mu_prior = nn.Parameter(torch.Tensor(1, latent_dim), requires_grad=False)
        self.mu_prior.data.fill_(0)

        self.logvar_prior = nn.Parameter(torch.Tensor(1, latent_dim), requires_grad=False)
        self.logvar_prior.data.fill_(0)

        self.logvar_uniform_prior = nn.Parameter(torch.Tensor(1, latent_dim), requires_grad=False)
        self.logvar_uniform_prior.data.fill_(10)

        self.encoder_old = Encoder(hidden_dim, latent_dim, input_dim)
        self.encoder_old.requires_grad_(False)

    def forward(self, x, z):
        post_mu, post_logvar = self.encoder_old(x, 0)

        stnd_prior = log_norm_pdf(z, self.mu_prior, self.logvar_prior)
        post_prior = log_norm_pdf(z, post_mu, post_logvar)
        unif_prior = log_norm_pdf(z, self.mu_prior, self.logvar_uniform_prior)

        gaussians = [stnd_prior, post_prior, unif_prior]
        gaussians = [g.add(np.log(w)) for g, w in zip(gaussians, self.mixture_weights)]

        density_per_gaussian = torch.stack(gaussians, dim=-1)

        return torch.logsumexp(density_per_gaussian, dim=-1)


class Encoder(nn.Module):
    def __init__(self, hidden_dim, latent_dim, input_dim, eps=1e-1):
        super(Encoder, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.ln4 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.ln5 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, dropout_rate):
        norm = x.pow(2).sum(dim=-1).sqrt()
        x = x / norm[:, None]

        x = F.dropout(x, p=dropout_rate, training=self.training)

        h1 = self.ln1(swish(self.fc1(x)))
        h2 = self.ln2(swish(self.fc2(h1) + h1))
        h3 = self.ln3(swish(self.fc3(h2) + h1 + h2))
        h4 = self.ln4(swish(self.fc4(h3) + h1 + h2 + h3))
        h5 = self.ln5(swish(self.fc5(h4) + h1 + h2 + h3 + h4))
        return self.fc_mu(h5), self.fc_logvar(h5)


class VAE(nn.Module):
    def __init__(self, hidden_dim, latent_dim, input_dim):
        super(VAE, self).__init__()

        self.encoder = Encoder(hidden_dim, latent_dim, input_dim)
        self.prior = CompositePrior(hidden_dim, latent_dim, input_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, user_ratings, beta=None, gamma=1, dropout_rate=0.5, calculate_loss=True):
        mu, logvar = self.encoder(user_ratings, dropout_rate=dropout_rate)
        z = self.reparameterize(mu, logvar)
        x_pred = self.decoder(z)

        if calculate_loss:
            if gamma:
                norm = user_ratings.sum(dim=-1)
                kl_weight = gamma * norm
            elif beta:
                kl_weight = beta

            mll = (F.log_softmax(x_pred, dim=-1) * user_ratings).sum(dim=-1).mean()
            kld = (log_norm_pdf(z, mu, logvar) - self.prior(user_ratings, z)).sum(dim=-1).mul(kl_weight).mean()
            negative_elbo = -(mll - kld)

            return (mll, kld), negative_elbo

        else:
            return x_pred

    def update_prior(self):
        self.prior.encoder_old.load_state_dict(deepcopy(self.encoder.state_dict()))


class Batch:
    def __init__(self, device, idx, data_in, data_out=None):
        self._device = device
        self._idx = idx
        self._data_in = data_in
        self._data_out = data_out

    def get_idx(self):
        return self._idx

    def get_idx_to_dev(self):
        return torch.LongTensor(self.get_idx()).to(self._device)

    def get_ratings(self, is_out=False):
        data = self._data_out if is_out else self._data_in
        return data[self._idx]

    def get_ratings_to_dev(self, is_out=False):
        return torch.Tensor(
            self.get_ratings(is_out).toarray()
        ).to(self._device)
