import glob
import os.path

from imgutils.data import load_image
from torch.utils.data import Dataset


class LocalImageDataset(Dataset):
    def __init__(self, image_dir, labels, transform=None):
        self.labels = labels
        self._label_map = {l: i for i, l in enumerate(labels)}

        self.images = []
        for lid, label in enumerate(labels):
            for f in glob.glob(os.path.join(image_dir, label, '*')):
                self.images.append((f, lid))

        self.transform = transform

        self._cached_images = {}

    def _getitem(self, index):
        if index not in self._cached_images:
            image_file, lid = self.images[index]
            image = load_image(image_file, force_background='white', mode='RGB')
            self._cached_images[index] = (image, lid)
        else:
            image, lid = self._cached_images[index]

        return image, lid

    def __getitem__(self, index):
        image, lid = self._getitem(index)
        if self.transform is not None:
            image = self.transform(image)
        return image, lid

    def __len__(self):
        return len(self.images)
