from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import register_loss, LossReduction


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, num_classes: int, smoothing: float = 0.1,
                 reduction: Literal['mean', 'sum'] = 'mean', weight=None):
        super(LabelSmoothingCrossEntropy, self).__init__()
        _ = num_classes
        self.smoothing = smoothing
        weight = torch.as_tensor(weight).float() if weight is not None else weight
        self.register_buffer('weight', weight)
        self.weight: torch.Tensor
        self.loss_reduction = LossReduction(reduction)

    def forward(self, input, target):
        log_probs = F.log_softmax(input, dim=-1)

        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(self.smoothing / (input.size(-1) - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), 1.0 - self.smoothing)

        loss = torch.sum(-true_dist * log_probs, dim=-1)
        if self.weight is not None:
            wl = self.weight[target]
            loss = loss * wl

        return self.loss_reduction(loss)


register_loss('lsce', LabelSmoothingCrossEntropy)
