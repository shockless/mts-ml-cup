import bisect

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
    return bisect.bisect_left([18, 25, 35, 45, 55, 65], x)

def save_model(model, folder, filename):
    torch.save(model, f"{folder}/{filename}.pt")