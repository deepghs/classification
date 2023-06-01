from typing import Mapping, Tuple, Type, Dict, List, Optional

try:
    from typing import Literal
except (ImportError, ModuleNotFoundError):
    from typing_extensions import Literal

import torch.nn


class WeightAttachment(torch.nn.Module):
    def __init__(self, num_classes, weight=None):
        torch.nn.Module.__init__(self)
        self.num_classes = num_classes

        if weight is None:
            weight = torch.ones(num_classes, dtype=torch.float)
        self.register_buffer('weights', torch.tensor(weight))
        self.weights: Optional[torch.Tensor]

    def forward(self, loss, labels):
        return loss * self.weights[labels].to(loss.device)


class LossReduction(torch.nn.Module):
    def __init__(self, reduction: Literal['mean', 'sum'] = 'mean'):
        torch.nn.Module.__init__(self)
        if reduction not in {'mean', 'sum'}:
            raise ValueError(f'Unknown reduction type - {reduction!r}.')
        self.reduction = reduction

    def forward(self, loss):
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            assert False, 'Should not reach here!'


_KNOWN_LOSSES: Dict[str, Tuple[Type[torch.nn.Module], Tuple, Mapping]] = {}


def register_loss(name: str, module: Type[torch.nn.Module], *args, **kwargs):
    if name not in _KNOWN_LOSSES:
        _KNOWN_LOSSES[name] = (module, args, kwargs)
    else:
        raise ValueError(f'Loss {name!r} already exist!')


def get_loss_fn(name: str, num_classes: int, weight, reduction: Literal['mean', 'sum'] = 'mean'):
    if name in _KNOWN_LOSSES:
        module, args, kwargs = _KNOWN_LOSSES[name]

        # noinspection PyArgumentList
        return module(num_classes, *args, reduction=reduction, weight=weight, **kwargs)
    else:
        raise ValueError(f'Loss function should be one of the {list_losses()!r}, but {name!r} found.')


def list_losses() -> List[str]:
    return sorted(_KNOWN_LOSSES.keys())
