from typing import List

import numpy as np
from hbutils.random import keep_global_state, global_seed
from torch.utils.data import random_split, Dataset


class WrappedImageDataset(Dataset):
    def __init__(self, dataset, *transforms):
        self.dataset = dataset
        if not transforms:
            self.transforms = [lambda x: x]
        else:
            self.transforms = list(transforms)

    def __getitem__(self, item):
        data, label = self.dataset[item]
        return *(m(data) for m in self.transforms), label

    def __len__(self):
        return len(self.dataset)


@keep_global_state()
def dataset_split(dataset, ratios: List[float], seed: int = 0):
    global_seed(seed)
    counts = (np.array(ratios) * len(dataset)).astype(int)
    counts[-1] = len(dataset) - counts[:-1].sum()
    assert counts.sum() == len(dataset)
    return random_split(dataset, counts)
