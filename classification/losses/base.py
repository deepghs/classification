from typing import Mapping, Tuple, Type, Dict, List

try:
    from typing import Literal
except (ImportError, ModuleNotFoundError):
    from typing_extensions import Literal
import torch.nn

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
