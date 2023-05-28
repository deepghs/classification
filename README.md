# classification

## Let's Training the Classifier

```python
import math

from ditk import logging
from torchvision import transforms

from classification.dataset import LocalImageDataset, dataset_split, WrappedImageDataset
from classification.train import train_simple

logging.try_init_root(logging.INFO)

# Meta information for task
LABELS = ['monochrome', 'normal']  # labels of each class
WEIGHTS = [math.e ** 2, 1.0]  # weight of each class

# dataset directory (use your own, like the following)
# <dataset_dir>
# ├── class1
# │   ├── image1.jpg
# │   └── image2.png  # all readable format by PIL is okay
# ├── class2
# │   ├── image3.jpg
# │   └── image4.jpg
# └── class3
#     ├── image5.jpg
#     └── image6.jpg
DATASET_DIR = '/my/dataset/directory'

# data argument and preprocessing for train dataset
TRANSFORM_TRAIN = transforms.Compose([
    # data argumentation
    transforms.Resize(450),
    transforms.RandomCrop(400, padding=50, pad_if_needed=True, padding_mode='reflect'),
    transforms.RandomRotation((-180, 180)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(0.10, 0.10, 0.10, 0.10),

    # preprocessing (recommended to be the same as tests)
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# preprocess the test dataset
TRANSFORM_TEST = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

# prepare the dataset
dataset = LocalImageDataset(DATASET_DIR, LABELS)
test_ratio = 0.2
train_dataset, test_dataset = dataset_split(dataset, [1 - test_ratio, test_ratio])
train_dataset = WrappedImageDataset(train_dataset, TRANSFORM_TRAIN)
test_dataset = WrappedImageDataset(test_dataset, TRANSFORM_TEST)

# Let's GO!
if __name__ == '__main__':
    train_simple(
        # work directory for training task
        # resume will be auto-performed after interruption
        workdir='runs/demo_exp',

        # all model in timm is usable, 
        # see supported models with timm.list_models() or see the performance table at 
        # https://github.com/huggingface/pytorch-image-models/blob/main/results/results-imagenet.csv
        model_name='mobilenetv3_large_100',

        # labels and weights, all 1 when weights not given
        labels=LABELS,
        loss_weight=WEIGHTS,

        # datasets
        train_dataset=train_dataset,
        test_dataset=test_dataset,

        # train settings
        max_epochs=500,
        num_workers=8,
        eval_epoch=5,

        # hyper-parameters
        batch_size=16,
        learning_rate=0.001,
        weight_decay=1e-3,

        # seed
        seed=0
    )
```