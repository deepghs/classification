from typing import List

import numpy as np
from hbutils.random import keep_global_state, global_seed
from torch.utils.data import random_split, Dataset


class WrappedImageDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, item):
        data, label = self.dataset[item]
        if self.transform is not None:
            data = self.transform(data)
        return data, label

    def __len__(self):
        return len(self.dataset)


@keep_global_state()
def dataset_split(dataset, ratios: List[float], seed: int = 0):
    global_seed(seed)
    counts = (np.array(ratios) * len(dataset)).astype(int)
    counts[-1] = len(dataset) - counts[:-1].sum()
    assert counts.sum() == len(dataset)
    return random_split(dataset, counts)
