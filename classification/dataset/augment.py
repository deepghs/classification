import random
from functools import wraps
from typing import Tuple, Callable

from PIL import Image
from torch import nn
from torchvision.transforms import RandomCrop


def prob_op(prob, op):
    @wraps(op)
    def _func(img):
        if random.random() < prob:
            return op(img)
        else:
            return img

    return _func


def _to_greyscale(img):
    origin_mode = img.mode
    return img.convert('L').convert(origin_mode)


def prob_greyscale(prob: float = 0.5):
    return prob_op(prob, _to_greyscale)


class RangeRandomCrop(nn.Module):
    def __init__(self, sizes: Tuple[int, int], padding=None, pad_if_needed=False, fill=0, padding_mode="constant"):
        nn.Module.__init__(self)
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

        self.min_size, self.max_size = sizes

    def _get_random_size(self):
        return random.randint(self.min_size, self.max_size)

    def forward(self, img):
        crop = RandomCrop(self._get_random_size(), self.padding, self.pad_if_needed, self.fill, self.padding_mode)
        return crop(img)


def _fn_min_center_crop(image: Image.Image) -> Image.Image:
    size = min(image.width, image.height)
    left = (image.width - size) // 2
    top = (image.height - size) // 2
    right, bottom = left + size, top + size
    return image.crop((left, top, right, bottom))


def min_center_crop() -> Callable[[Image.Image], Image]:
    return _fn_min_center_crop
