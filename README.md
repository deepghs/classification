# classification

## Installation

```shell
pip install -r requirements.txt
pip install -r requirements-onnx.txt
```

## Let's Training the Classifier

Save this to `train.py`

```python
import math

from ditk import logging
from torchvision import transforms

from classification.dataset import LocalImageDataset, dataset_split, WrappedImageDataset, RangeRandomCrop, prob_greyscale
from classification.train import train_simple

logging.try_init_root(logging.INFO)

# Meta information for task
LABELS = ['monochrome', 'normal']  # labels of each class
WEIGHTS = [math.e ** 2, 1.0]  # weight of each class
assert len(LABELS) == len(WEIGHTS), \
    f'Labels and weights should have the same length, but {len(LABELS)}(labels) and {len(WEIGHTS)}(weights) found.'

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

# data augment and preprocessing for train dataset
TRANSFORM_TRAIN = transforms.Compose([
    # data augmentation
    # prob_greyscale(0.5),  # use this line when color is not important
    transforms.Resize((500, 500)),
    RangeRandomCrop((400, 500), padding=0, pad_if_needed=True, padding_mode='reflect'),
    transforms.RandomRotation((-45, 45)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.10, 0.10, 0.05, 0.03),

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
# dataset for visualization (it may be slow when there are too many classes)
# if you do not need this, just comment it
TRANSFORM_VISUAL = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
])

# prepare the dataset
# disable cache when training on large dataset
dataset = LocalImageDataset(DATASET_DIR, LABELS, no_cache=True)
test_ratio = 0.2
train_dataset, test_dataset = dataset_split(dataset, [1 - test_ratio, test_ratio])
train_dataset = WrappedImageDataset(train_dataset, TRANSFORM_TRAIN)
test_dataset = WrappedImageDataset(test_dataset, TRANSFORM_TEST, TRANSFORM_VISUAL)
# if you do not need visualization, just use this
# test_dataset = WrappedImageDataset(test_dataset, TRANSFORM_TEST)

# Let's GO!
if __name__ == '__main__':
    train_simple(
        # work directory for training task
        # resume will be auto-performed after interruption
        workdir='runs/demo_exp',

        # all model in timm is usable, 
        # see supported models with timm.list_models() or see the performance table at 
        # https://github.com/huggingface/pytorch-image-models/blob/main/results/results-imagenet.csv
        # Recommendation:
        # 1. use caformer_s36.sail_in22k_ft_in1k_384 for training
        # 2. use mobilenetv3_large_100 for distillation
        model_name='caformer_s36.sail_in22k_ft_in1k_384',

        # labels and weights, all 1 when weights not given
        labels=LABELS,
        loss_weight=WEIGHTS,

        # datasets
        train_dataset=train_dataset,
        test_dataset=test_dataset,

        # train settings, pretrained model will be used
        max_epochs=100,
        num_workers=8,
        eval_epoch=1,
        key_metric='accuracy',
        loss='focal',  # use `sce` when the dataset is not guaranteed to be cleaned
        seed=0,
        drop_path_rate=0.4,  # use this when training on caformer

        # hyper-parameters
        batch_size=16,
        learning_rate=1e-5,  # 1e-5 recommended for caformer's fine-tuning
        weight_decay=1e-3,
    )
```

And run

```
accelerate launch train.py
```

## Distillate the Model

Save this to `dist.py`

```python
from ditk import logging
from torchvision import transforms

from classification.dataset import LocalImageDataset, dataset_split, WrappedImageDataset, RangeRandomCrop, prob_greyscale
from classification.train import train_distillation

logging.try_init_root(logging.INFO)

# Meta information for task
LABELS = ['monochrome', 'normal']  # labels of each class
WEIGHTS = [1.0, 1.0]  # weight of each class
assert len(LABELS) == len(WEIGHTS), \
    f'Labels and weights should have the same length, but {len(LABELS)}(labels) and {len(WEIGHTS)}(weights) found.'

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

# data augment and preprocessing for train dataset
TRANSFORM_TRAIN = transforms.Compose([
    # data augmentation
    # prob_greyscale(0.5),  # use this line when color is not important
    transforms.Resize((500, 500)),
    RangeRandomCrop((400, 500), padding=0, pad_if_needed=True, padding_mode='reflect'),
    transforms.RandomRotation((-45, 45)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.10, 0.10, 0.05, 0.03),

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
# dataset for visualization (it may be slow when there are too many classes)
# if you do not need this, just comment it
TRANSFORM_VISUAL = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
])

# prepare the dataset
# disable cache when training on large dataset
dataset = LocalImageDataset(DATASET_DIR, LABELS, no_cache=True)
test_ratio = 0.2
train_dataset, test_dataset = dataset_split(dataset, [1 - test_ratio, test_ratio])
train_dataset = WrappedImageDataset(train_dataset, TRANSFORM_TRAIN)
test_dataset = WrappedImageDataset(test_dataset, TRANSFORM_TEST, TRANSFORM_VISUAL)
# if you do not need visualization, just use this
# test_dataset = WrappedImageDataset(test_dataset, TRANSFORM_TEST)

# Let's GO!
if __name__ == '__main__':
    train_distillation(
        # work directory for student model of distillation task
        # resume will be auto-performed after interruption
        workdir='runs/demo_exp_dist',

        # all model in timm is usable, 
        # see supported models with timm.list_models() or see the performance table at 
        # https://github.com/huggingface/pytorch-image-models/blob/main/results/results-imagenet.csv
        # Recommendation:
        # 1. use caformer_s36.sail_in22k_ft_in1k_384 for training
        # 2. use mobilenetv3_large_100 for distillation
        model_name='mobilenetv3_large_100',

        # distillation from the teacher model in runs/demo_exp
        teacher_workdir='runs/demo_exp',

        # labels and weights, all 1 when weights not given
        labels=LABELS,
        loss_weight=WEIGHTS,

        # datasets
        train_dataset=train_dataset,
        test_dataset=test_dataset,

        # train settings, pretrained model will be used
        max_epochs=500,
        num_workers=8,
        eval_epoch=5,
        key_metric='accuracy',
        loss='focal',  # use `sce` when the dataset is not guaranteed to be cleaned
        seed=0,
        # drop_path_rate=0.4,  # use this when training on caformer

        # distillation settings
        temperature=7.0,
        alpha=0.3,

        # hyper-parameters
        batch_size=16,
        learning_rate=1e-4,  # 1e-5 recommended for caformer's fine-tuning
        weight_decay=1e-3,
    )

```

And then run

```shell
accelerate launch dist.py
```

## Export onnx model

### From checkpoint

Attention that this export is only supported for ckpt dumped with `classficiation` module, for it contains extra
information except state dict of the network.

```shell
python -m classification.onnx export --help
```

```text
Usage: python -m classification.onnx export [OPTIONS]

  Export existing checkpoint to onnx

Options:
  -i, --input FILE       Input checkpoint to export.  [required]
  -s, --imgsize INTEGER  Image size for input.  [default: 384]
  -D, --non-dynamic      Do not export model with dynamic input height and
                         width.
  -V, --verbose          Show verbose information.
  -o, --output FILE      Output file of onnx model
  -h, --help             Show this message and exit.
```

## From training work directory

```shell
python -m classification.onnx dump -w runs/demo_exp
```

Then the `runs/demo_exp/ckpts/best.ckpt` will be dumped to `runs/demo_exp/onnxs/best.onnx`

Here is the help information

```text
Usage: python -m classification.onnx dump [OPTIONS]

  Dump onnx model from existing work directory.

Options:
  -w, --workdir DIRECTORY  Work directory of the training.  [required]
  -s, --imgsize INTEGER    Image size for input.  [default: 384]
  -D, --non-dynamic        Do not export model with dynamic input height and
                           width.
  -V, --verbose            Show verbose information.
  -h, --help               Show this message and exit.
```

## Publish Trained Models

Before start, set the `HF_TOKEN` variable to your huggingface token

```shell
# on Linux
export HF_TOKEN=xxxxxxxx
```

Publish trained models (including ckpt, onnx, metrics data and figures) to huggingface
repository `your/huggingface_repo`

```shell
python -m classification.publish huggingface -w runs/your_model_dir -n name_of_the_model -r your/huggingface_repo
```

List all the models in given repository `your/huggingface_repo`, which can be used in README

```shell
python -m classification.list huggingface -r your/huggingface_repo
```

An example model repository: https://huggingface.co/deepghs/anime_style_ages
