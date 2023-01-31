from torch import nn

from utils import torch_age_bucket


class RegressionCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss = nn.CrossEntropyLoss()

    def forward(self, output, target):
        age_buckets = torch_age_bucket(output)
        loss = self.loss(age_buckets, target)

        return loss
