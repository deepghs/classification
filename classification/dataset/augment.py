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

    def _get_random_size(self, width, height):
        if isinstance(self.min_size, int):
            min_width = min_height = self.min_size
        else:
            min_width = int(width * self.min_size)
            min_height = int(height * self.min_size)
        if isinstance(self.max_size, int):
            max_width = max_height = self.max_size
        else:
            max_width = int(width * self.max_size)
            max_height = int(height * self.max_size)
        return random.randint(min_height, max_height), random.randint(min_width, max_width)

    def forward(self, img):
        if isinstance(img, Image.Image):
            width, height = img.width, img.height
        else:
            height, width = img.shape[-2:]
        crop = RandomCrop(
            self._get_random_size(width, height),
            padding=self.padding,
            pad_if_needed=self.pad_if_needed,
            fill=self.fill,
            padding_mode=self.padding_mode
        )
        return crop(img)


def _fn_min_center_crop(image: Image.Image) -> Image.Image:
    size = min(image.width, image.height)
    left = (image.width - size) // 2
    top = (image.height - size) // 2
    right, bottom = left + size, top + size
    return image.crop((left, top, right, bottom))


def min_center_crop() -> Callable[[Image.Image], Image.Image]:
    return _fn_min_center_crop
