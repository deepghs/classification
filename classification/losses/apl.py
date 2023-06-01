"""
    Based on paper: https://proceedings.mlr.press/v119/ma20c.html
    PyTorch Implement: https://github.com/HanxunH/Active-Passive-Losses/blob/master/loss.py
"""

try:
    from typing import Literal
except (ImportError, ModuleNotFoundError):
    from typing_extensions import Literal

import torch
import torch.nn.functional as F

from .base import register_loss, WeightAttachment, LossReduction


class NormalizedCrossEntropy(torch.nn.Module):
    def __init__(self, num_classes, scale=1.0,
                 reduction: Literal['mean', 'sum'] = 'mean', weight=None):
        torch.nn.Module.__init__(self)
        self.num_classes = num_classes
        self.scale = scale
        self.cls_weight = WeightAttachment(num_classes, weight)
        self.loss_reduction = LossReduction(reduction)

    def forward(self, logits, labels):
        score = F.log_softmax(logits, dim=1)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(score.device)
        nce = -1 * torch.sum(label_one_hot * score, dim=1) / (- score.sum(dim=1))

        nce = self.cls_weight(nce, labels)
        return self.scale * self.loss_reduction(nce)


class NormalizedFocalLoss(torch.nn.Module):
    def __init__(self, num_classes, scale=1.0, gamma=0.0,
                 reduction: Literal['mean', 'sum'] = 'mean', weight=None):
        torch.nn.Module.__init__(self)
        self.gamma = gamma
        self.num_classes = num_classes
        self.scale = scale
        self.cls_weight = WeightAttachment(num_classes, weight)
        self.loss_reduction = LossReduction(reduction)

    def forward(self, logits, labels):
        logpt = F.log_softmax(logits, dim=1)
        normalizor = torch.sum(-1 * (1 - logpt.data.exp()) ** self.gamma * logpt, dim=1)
        logpt = logpt.gather(1, labels.view(-1, 1))
        logpt = logpt.view(-1)

        pt = torch.autograd.Variable(logpt.data.exp())
        loss = -1 * (1 - pt) ** self.gamma * logpt
        loss = self.scale * loss / normalizor

        loss = self.cls_weight(loss, labels)
        return self.loss_reduction(loss)


class MeanAbsoluteError(torch.nn.Module):
    def __init__(self, num_classes, scale=1.0,
                 reduction: Literal['mean', 'sum'] = 'mean', weight=None):
        torch.nn.Module.__init__(self)
        self.num_classes = num_classes
        self.scale = scale
        self.cls_weight = WeightAttachment(num_classes, weight)
        self.loss_reduction = LossReduction(reduction)

    def forward(self, logits, labels):
        score = F.softmax(logits, dim=1)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(score.device)
        mae = 1. - torch.sum(label_one_hot * score, dim=1)
        # Note: Reduced MAE
        # Original: torch.abs(pred - label_one_hot).sum(dim=1)
        # $MAE = \sum_{k=1}^{K} |\bm{p}(k|\bm{x}) - \bm{q}(k|\bm{x})|$
        # $MAE = \sum_{k=1}^{K}\bm{p}(k|\bm{x}) - p(y|\bm{x}) + (1 - p(y|\bm{x}))$
        # $MAE = 2 - 2p(y|\bm{x})$

        mae = self.cls_weight(mae, labels)
        return self.scale * self.loss_reduction(mae)


class ReverseCrossEntropy(torch.nn.Module):
    def __init__(self, num_classes, scale=1.0,
                 reduction: Literal['mean', 'sum'] = 'mean', weight=None):
        torch.nn.Module.__init__(self)
        self.num_classes = num_classes
        self.scale = scale
        self.cls_weight = WeightAttachment(num_classes, weight)
        self.loss_reduction = LossReduction(reduction)

    def forward(self, logits, labels):
        score = F.softmax(logits, dim=1)
        score = torch.clamp(score, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(score.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1 * torch.sum(score * torch.log(label_one_hot), dim=1))

        rce = self.cls_weight(rce, labels)
        return self.scale * self.loss_reduction(rce)


class NCEAndRCE(torch.nn.Module):
    def __init__(self, num_classes, alpha=1.0, beta=1.0,
                 reduction: Literal['mean', 'sum'] = 'mean', weight=None):
        torch.nn.Module.__init__(self)
        self.num_classes = num_classes
        self.nce = NormalizedCrossEntropy(
            scale=alpha, num_classes=num_classes,
            reduction=reduction, weight=weight,
        )
        self.rce = ReverseCrossEntropy(
            scale=beta, num_classes=num_classes,
            reduction=reduction, weight=weight,
        )

    def forward(self, logits, labels):
        return self.nce(logits, labels) + self.rce(logits, labels)


register_loss('nce+rce', NCEAndRCE)


class NCEAndMAE(torch.nn.Module):
    def __init__(self, num_classes, alpha=1.0, beta=1.0,
                 reduction: Literal['mean', 'sum'] = 'mean', weight=None):
        torch.nn.Module.__init__(self)
        self.num_classes = num_classes
        self.nce = NormalizedCrossEntropy(
            scale=alpha, num_classes=num_classes,
            reduction=reduction, weight=weight,
        )
        self.mae = MeanAbsoluteError(
            scale=beta, num_classes=num_classes,
            reduction=reduction, weight=weight,
        )

    def forward(self, logits, labels):
        return self.nce(logits, labels) + self.mae(logits, labels)


register_loss('nce+mae', NCEAndMAE)


class NFLAndRCE(torch.nn.Module):
    def __init__(self, num_classes, alpha=1.0, beta=1.0, gamma=0.5,
                 reduction: Literal['mean', 'sum'] = 'mean', weight=None):
        torch.nn.Module.__init__(self)
        self.num_classes = num_classes
        self.nfl = NormalizedFocalLoss(
            scale=alpha, gamma=gamma, num_classes=num_classes,
            reduction=reduction, weight=weight,
        )
        self.rce = ReverseCrossEntropy(
            scale=beta, num_classes=num_classes,
            reduction=reduction, weight=weight,
        )

    def forward(self, logits, labels):
        return self.nfl(logits, labels) + self.rce(logits, labels)


register_loss('nfl+rce', NFLAndRCE)


class NFLAndMAE(torch.nn.Module):
    def __init__(self, num_classes, alpha=1.0, beta=1.0, gamma=0.5,
                 reduction: Literal['mean', 'sum'] = 'mean', weight=None):
        torch.nn.Module.__init__(self)
        self.num_classes = num_classes
        self.nfl = NormalizedFocalLoss(
            scale=alpha, gamma=gamma, num_classes=num_classes,
            reduction=reduction, weight=weight,
        )
        self.mae = MeanAbsoluteError(
            scale=beta, num_classes=num_classes,
            reduction=reduction, weight=weight,
        )

    def forward(self, logits, labels):
        return self.nfl(logits, labels) + self.mae(logits, labels)


register_loss('nfl+mae', NFLAndMAE)
