import torch
from torch import nn
from torch.nn import functional as F

from .base import register_loss


class FocalLoss(nn.Module):
    """
    Based on https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/8
    """

    def __init__(self, num_classes, gamma=2., reduction='mean', weight=None):
        nn.Module.__init__(self)
        self.num_classes = num_classes
        weight = torch.as_tensor(weight).float() if weight is not None else weight
        self.register_buffer('weight', weight)
        self.weight: torch.Tensor

        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, labels):
        log_prob = F.log_softmax(logits, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            labels,
            weight=self.weight,
            reduction=self.reduction
        )


register_loss('focal', FocalLoss)
