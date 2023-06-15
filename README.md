# classification

## Let's Training the Classifier

```python
import math

from ditk import logging
from torchvision import transforms

from classification.dataset import LocalImageDataset, dataset_split, WrappedImageDataset, RangeRandomCrop
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

# prepare the dataset
# disable cache when training on large dataset
dataset = LocalImageDataset(DATASET_DIR, LABELS, no_cache=True)
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

        # train settings
        max_epochs=500,
        num_workers=8,
        eval_epoch=5,
        key_metric='accuracy',
        loss='focal',  # use `sce` when the dataset is guaranteed to be cleaned
        seed=0,
        # drop_path_rate=0.4,  # use this when training on caformer

        # hyper-parameters
        batch_size=16,
        learning_rate=1e-3,  # 5e-5 recommended for caformer, 1e-5 for fine-tuning
        weight_decay=1e-3,
    )
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