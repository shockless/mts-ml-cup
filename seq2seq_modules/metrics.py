import torch
# from torchmetrics import AUROC, F1Score
from sklearn.metrics import f1_score, roc_auc_score

from seq2seq_modules.utils import numpy_age_bucket


def GENDER_METRIC(logits, targets):
    return {"Gender GINI": 2 * roc_auc_score(targets.numpy(), torch.sigmoid(logits).numpy()[:, 1]) - 1}


def AGE_METRIC_REGRESSION(logits, targets):
    return {"Age F1": f1_score(numpy_age_bucket(targets.numpy()), logits.argmax(dim=1).numpy(), average="weighted")}


def AGE_METRIC(logits, targets):
    return {"Age F1": f1_score(targets.numpy(), logits.argmax(dim=1).numpy(), average="weighted")}
