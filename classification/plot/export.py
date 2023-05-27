import os.path

import numpy as np
import torch
from PIL import Image
from hbutils.system import TemporaryDirectory
from matplotlib import pyplot as plt

try:
    from typing import Literal
except (ImportError, ModuleNotFoundError):
    from typing_extensions import Literal


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    elif isinstance(x, np.ndarray):
        return x
    elif isinstance(x, dict):
        return type(x)({key: _to_numpy(value) for key, value in x.items()})
    elif isinstance(x, (list, tuple)):
        return type(x)([_to_numpy(item) for item in x])
    else:
        return x


def plt_export(func, *args, figsize=(6, 6), **kwargs) -> Image.Image:
    fig = plt.Figure(figsize=figsize)
    fig.tight_layout()
    func(fig.gca(), *_to_numpy(args), **_to_numpy(kwargs))

    with TemporaryDirectory() as td:
        imgfile = os.path.join(td, 'image.png')
        fig.savefig(imgfile)

        image = Image.open(imgfile)
        image.load()
        image = image.convert('RGB')
        return image
