import io
import os.path
from typing import Union, Mapping, Optional, List, Type

import numpy as np
import torch
from PIL import Image
from ditk import logging
from torch.utils.tensorboard import SummaryWriter

from ..models import load_model_from_ckpt, save_model_to_ckpt


class BaseLogger:
    def __init__(self, workdir, **kwargs):
        _ = kwargs
        self.workdir = workdir

    def tb_train_log(self, global_step, metrics: Mapping[str, Union[float, Image.Image]]):
        raise NotImplementedError

    def tb_eval_log(self, global_step, model, metrics: Mapping[str, Union[float, Image.Image]]):
        raise NotImplementedError


class TensorboardLogger(BaseLogger):
    def __init__(self, workdir, **kwargs):
        BaseLogger.__init__(self, workdir, **kwargs)
        self.tb_writer = SummaryWriter(workdir)

    def tb_log(self, global_step, data: Mapping[str, Union[float, Image.Image]]):
        logging_metrics = {}
        for key, value in data.items():
            if isinstance(value, (int, float)) or \
                    (isinstance(value, (torch.Tensor, np.ndarray)) and not value.shape):
                self.tb_writer.add_scalar(key, value, global_step)
                logging_metrics[key] = value
            elif isinstance(value, (torch.Tensor, np.ndarray)):
                self.tb_writer.add_image(key, value, global_step)
            elif isinstance(value, (Image.Image,)):
                # noinspection PyTypeChecker
                img_array = np.asarray(value.convert('RGB')).transpose((2, 0, 1))
                print(img_array.shape)
                self.tb_writer.add_image(key, img_array, global_step, dataformats='CHW')
            else:
                raise TypeError(f'Unknown data type for {key!r}: {value!r}')

        if logging_metrics:
            with io.StringIO() as sf:
                print(f'Metrics logging at epoch {global_step}', file=sf, end='')
                for key, value in sorted(logging_metrics.items()):
                    print(f', {key}: {value:.3f}', file=sf, end='')

                logging.info(sf.getvalue())

    def tb_train_log(self, global_step, metrics: Mapping[str, Union[float, Image.Image]]):
        self.tb_log(
            global_step,
            {f'train/{key}': value for key, value in metrics.items()}
        )

    def tb_eval_log(self, global_step, model, metrics: Mapping[str, Union[float, Image.Image]]):
        _ = model
        self.tb_log(
            global_step,
            {f'eval/{key}': value for key, value in metrics.items()}
        )


class CkptLogger(BaseLogger):
    def __init__(self, workdir: str, key_metric: str = 'accuracy', **kwargs):
        BaseLogger.__init__(self, workdir, **kwargs)
        self.key_metric = key_metric

        self.ckpt_dir = os.path.join(self.workdir, 'ckpts')
        self._last_step: Optional[int] = None
        self._best_metric_value: Optional[float] = None
        self._load_last()
        self._load_best()

    @property
    def _last_ckpt(self):
        return os.path.join(self.ckpt_dir, 'last.ckpt')

    def _load_last(self):
        if os.path.exists(self._last_ckpt):
            model = load_model_from_ckpt(self._last_ckpt)
            info = model.__info__
            self._last_step = info['step']
            logging.info(f'Last ckpt found at {self._last_step}, with previous step {self._last_step}')
        else:
            self._last_step = None
            logging.info('No last ckpt found.')

    def _save_last(self, global_step, model, metrics: Mapping[str, Union[float, Image.Image]]):
        model.__info__ = {**metrics, 'step': global_step}
        save_model_to_ckpt(model, self._last_ckpt)
        self._last_step = global_step
        logging.info(f'Last ckpt model epoch {global_step} saved')

    @property
    def _best_ckpt(self):
        return os.path.join(self.ckpt_dir, 'best.ckpt')

    def _load_best(self):
        if os.path.exists(self._best_ckpt):
            model = load_model_from_ckpt(self._best_ckpt)
            info = model.__info__
            self._best_metric_value = info[self.key_metric]
            step = info['step']
            logging.info(f'Best ckpt found at {self._best_ckpt}, '
                         f'with step {step} and {self.key_metric} {self._best_metric_value:.3f}')
        else:
            self._best_metric_value = None
            logging.info('No best ckpt found.')

    def _save_best(self, global_step, model, metrics: Mapping[str, Union[float, Image.Image]]):
        model.__info__ = {**metrics, 'step': global_step}
        if self._best_metric_value is None or model.__info__[self.key_metric] > self._best_metric_value:
            save_model_to_ckpt(model, self._best_ckpt)
            self._best_metric_value = model.__info__[self.key_metric]
            logging.info(f'Best ckpt model epoch {global_step} saved, '
                         f'with {self.key_metric}\'s new value {self._best_metric_value:.3f}')

    def tb_train_log(self, global_step, metrics: Mapping[str, Union[float, Image.Image]]):
        pass

    def tb_eval_log(self, global_step, model, metrics: Mapping[str, Union[float, Image.Image]]):
        self._save_last(global_step, model, metrics)
        self._save_best(global_step, model, metrics)


_DEFAULT_CLASSES: List[Type[BaseLogger]] = [TensorboardLogger, CkptLogger]


class TrainSession:
    def __init__(self, workdir: str, classes: Optional[List[Type[BaseLogger]]] = None, **kwargs):
        if classes is None:
            classes = _DEFAULT_CLASSES
        self.loggers = [cls(workdir, **kwargs) for cls in classes]

    def tb_train_log(self, global_step, metrics: Mapping[str, Union[float, Image.Image]]):
        for logger in self.loggers:
            logger.tb_train_log(global_step, metrics)

    def tb_eval_log(self, global_step, model, metrics: Mapping[str, Union[float, Image.Image]]):
        for logger in self.loggers:
            logger.tb_eval_log(global_step, model, metrics)


def _load_last_ckpt(workdir):
    ckpt_file = os.path.join(workdir, 'ckpts', 'last.ckpt')
    if os.path.exists(ckpt_file):
        model = load_model_from_ckpt(ckpt_file)
        arguments = model.__arguments__.copy()
        model_name = arguments.pop('name')
        labels = arguments.pop('labels')
        return model, model_name, labels, arguments
    else:
        return None, None, None, None
