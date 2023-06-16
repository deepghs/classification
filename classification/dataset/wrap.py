from torch.utils.data import Dataset


class TestDatasetWrapper(Dataset):
    def __init__(self, dataset):
        Dataset.__init__(self)
        self.dataset = dataset

    def __getitem__(self, item):
        data = self.dataset[item]
        if isinstance(data, tuple):
            return item, *data
        else:
            return item, data

    def __len__(self):
        return len(self.dataset)


class TestDatasetVisualWrapper(Dataset):
    def __init__(self, dataset):
        Dataset.__init__(self)
        self.dataset = dataset

    def __getitem__(self, item):
        input_, visual, *_, labels = self.dataset[item]
        return visual

    def __len__(self):
        return len(self.dataset)
