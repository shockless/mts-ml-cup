import bisect
import random

import numpy as np
import torch


def generate_square_subsequent_mask(sz):
    r"""
    Generate a square mask for the sequence. The masked positions are filled with float('-inf').
    Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask[0, :] = 0.0
    mask[:, 0] = 0.0
    return mask


def age_bucket(x):
    return bisect.bisect_left([25, 35, 45, 55, 65], x)


def torch_age_bucket(x):
    boundaries = torch.tensor([25, 35, 45, 55, 65])
    return torch.bucketize(x, boundaries, right=False)


def numpy_age_bucket(x):
    boundaries = np.array([25, 35, 45, 55, 65])
    return np.digitize(x, boundaries, right=True)


def save_model(model, folder, filename):
    torch.save(model, f"{folder}/{filename}.pt")


def fix_random_state(random_state: int = 42):
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)


def coles_collate_fn(data):
    batch_cat_features, batch_cont_features, batch_attention_mask, batch_target = data

    cat_features = torch.cat([part for part in batch_cat_features], dim=0)
    cont_features = torch.cat([part for part in batch_cont_features], dim=0)
    attention_mask = torch.cat([part for part in batch_attention_mask], dim=0)
    target = torch.cat([part for part in batch_target], dim=0)

    return cat_features, cont_features, attention_mask, target