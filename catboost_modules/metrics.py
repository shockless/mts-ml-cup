from sklearn.metrics import f1_score, roc_auc_score

from seq2seq_modules.utils import numpy_age_bucket


def AGE_METRIC_REGRESSION(logits, targets):
    return {"Age F1": f1_score(numpy_age_bucket(targets.to_numpy()), numpy_age_bucket(logits), average="weighted")}


def AGE_METRIC(logits, targets):
    return {"Age F1": f1_score(targets.to_numpy(), logits, average="weighted")}

def GENDER_METRIC(logits, targets):
    return {"Gender GINI": 2 * roc_auc_score(targets.to_numpy(), logits[:, 1]) - 1}
