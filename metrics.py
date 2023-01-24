from torchmetrics import AUROC, F1Score

from utils import age_bucket


def GENDER_METRIC(logits, targets):
    score = AUROC(task="binary")
    return {"Gender ROC AUC": score(logits, targets)}


def AGE_METRIC(logits, targets):
    score = F1Score(task="multiclass", average="macro", num_classes=7)
    return {"Age F1": score(age_bucket(logits), targets)}
