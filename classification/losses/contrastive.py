import torch
from torch import nn


class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        nn.Module.__init__(self)

        self.register_buffer('margin', torch.tensor(margin))
        self.margin: torch.Tensor

        self.register_buffer('_zero', torch.tensor(0.0))
        self._zero: torch.Tensor

    def forward(self, logits, labels):
        assert len(labels.shape) == 1 and len(logits.shape) == 2 and logits.shape[0] == labels.shape[0], \
            f'Expected to be logit(BxC) and labels(B), but logit({logits.shape}) and labels({logits.shape}) found.'

        batch = labels.shape[0]
        idx = torch.arange(batch)
        valid = (idx != idx.unsqueeze(-1)).to(self.margin.device)

        _same = labels == labels.unsqueeze(-1)
        same_mask = (_same & valid).type(logits.dtype)
        not_same_mask = (~_same & valid).type(logits.dtype)

        dists = torch.nn.functional.pairwise_distance(logits, logits.unsqueeze(-2))
        mdists = torch.maximum(self.margin - dists, self._zero.type(logits.dtype))

        return ((dists ** 2 * same_mask) + (mdists ** 2 * not_same_mask)).sum() / valid.sum()
