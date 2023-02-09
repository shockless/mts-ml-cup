# from torchmetrics import AUROC, F1Score
from sklearn.metrics import f1_score

from utils import age_bucket


# def GENDER_METRIC(logits, targets):
#     score = AUROC(task="binary")
#     return {"Gender GINI": 2 * score(logits, targets) - 1}


# def AGE_METRIC_REGRESSION(logits, targets):
#     score = F1Score(task="multiclass", average="macro", num_classes=7)
#     return {"Age F1": score(age_bucket(logits), targets)}


def AGE_METRIC(logits, targets):
    return {"Age F1": f1_score(targets.numpy(), logits.argmax(dim=1).numpy(), average="macro")}
