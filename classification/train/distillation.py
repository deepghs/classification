import os
import warnings
from typing import Optional, List

from torch import nn

try:
    from typing import Literal
except (ModuleNotFoundError, ImportError):
    from typing_extensions import Literal

import torch
from accelerate import Accelerator
from ditk import logging
from hbutils.random import global_seed
from sklearn.metrics import accuracy_score
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from .base import register_task_type, put_meta_at_workdir
from .metrics import cls_map_score, cls_auc_score
from .profile import torch_model_profile
from .session import _load_last_ckpt, TrainSession
from ..losses import get_loss_fn
from ..models import get_backbone_model
from ..plot import plt_export, plt_confusion_matrix, plt_pr_curve, plt_p_curve, plt_r_curve, plt_f1_curve, plt_roc_curve

_PRESET_KWARGS = {
    'pretrained': True,
}

_TASK_NAME = 'distillation'

register_task_type(
    _TASK_NAME,
    onnx_conf=dict(
        use_softmax=True,
    ),
)


def train_distillation(
        workdir: str, model_name: str, teacher_workdir: str, labels: List[str],
        train_dataset: Dataset, test_dataset: Dataset,
        batch_size: int = 16, max_epochs: int = 500, learning_rate: float = 0.001,
        weight_decay: float = 1e-3, num_workers: Optional[int] = 8, eval_epoch: int = 5,
        temperature: float = 7.0, alpha=0.3,
        key_metric: Literal['accuracy', 'AUC', 'mAP'] = 'accuracy', cls_loss='focal',
        loss_weight=None, seed: Optional[int] = 0, **model_args):
    if seed is not None:
        # native random, numpy, torch and faker's seeds are includes
        # if you need to register more library for seeding, see:
        # https://hansbug.github.io/hbutils/main/api_doc/random/state.html#register-random-source
        logging.info(f'Globally set the random seed {seed!r}.')
        global_seed(seed)

    os.makedirs(workdir, exist_ok=True)
    put_meta_at_workdir(workdir, _TASK_NAME)
    accelerator = Accelerator(
        # mixed_precision=self.cfgs.mixed_precision,
        step_scheduler_with_optimizer=False,
    )

    model_args = {**_PRESET_KWARGS, **model_args}
    model, _model_name, _labels, _model_args = _load_last_ckpt(workdir)
    if model:
        logging.info(f'Load previous ckpt from work directory {workdir!r}.')
        assert _model_name == model_name, \
            f'Resumed model name {_model_args!r} not match with specified name {model_name!r}.'
        assert _labels == labels, \
            f'Resumed labels {_labels!r} from student model not match with specified labels {labels!r}.'
        if _model_args != model_args:
            warnings.warn(f'Resumed model arguments {_model_args!r} '
                          f'not match with specified arguments {model_args!r}.')
        previous_epoch = model.__info__['step']
        logging.info(f'Previous step is {previous_epoch!r}')
    else:
        logging.info(f'No previous ckpt found, load new model {model_name!r} with {model_args!r}.')
        model = get_backbone_model(model_name, labels, **model_args)
        previous_epoch = 0

    t_model, _t_model_name, _t_labels, _model_args = _load_last_ckpt(teacher_workdir)
    if t_model:
        t_info = getattr(t_model, '__info__') or {}
        logging.info(f'Load teacher model from workdir {teacher_workdir!r}, with metrics {t_info!r}.')
        assert _t_labels == labels, \
            f'Resumed labels {_t_labels!r} from teacher model not match with specified labels {labels!r}.'
    else:
        raise FileNotFoundError(f'Teacher model not found in workdir {teacher_workdir!r}.')
    t_model.eval()

    sample_input, _ = test_dataset[0]
    torch_model_profile(model, sample_input.unsqueeze(0))  # profile the model

    num_workers = num_workers or min(os.cpu_count(), batch_size)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, drop_last=True
    )
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    if loss_weight is None:
        loss_weight = torch.ones(len(labels), dtype=torch.float)
    student_loss_fn = get_loss_fn(cls_loss, len(labels), loss_weight, reduction='mean')
    distillation_loss_fn = nn.KLDivLoss(reduction="batchmean")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = lr_scheduler.OneCycleLR(
        optimizer, max_lr=learning_rate,
        steps_per_epoch=len(train_dataloader), epochs=max_epochs,
        pct_start=0.15, final_div_factor=20.
    )

    # noinspection PyTypeChecker
    model, student_loss_fn, \
        t_model, distillation_loss_fn, \
        optimizer, train_dataloader, test_dataloader, scheduler = \
        accelerator.prepare(
            model, student_loss_fn,
            t_model, distillation_loss_fn,
            optimizer, train_dataloader, test_dataloader, scheduler
        )

    session = TrainSession(workdir, key_metric=key_metric)
    logging.info('Training start!')
    for epoch in range(previous_epoch + 1, max_epochs + 1):
        model.train()
        train_all_loss, train_distillation_loss, train_cls_loss = 0.0, 0.0, 0.0
        train_total = 0
        train_y_true, train_y_pred, train_y_score = [], [], []
        for i, (inputs, labels_) in enumerate(tqdm(train_dataloader)):
            inputs = inputs.float()
            inputs = inputs.to(accelerator.device)
            labels_ = labels_.to(accelerator.device)

            with torch.no_grad():
                teacher_outputs = t_model(inputs)

            optimizer.zero_grad()
            outputs = model(inputs)
            train_y_true.append(labels_)
            train_y_pred.append(outputs.argmax(dim=1))
            train_y_score.append(torch.softmax(outputs, dim=1))
            train_total += labels_.shape[0]

            cls_loss = student_loss_fn(outputs, labels_)
            distillation_loss = distillation_loss_fn(
                torch.softmax(outputs / temperature, dim=1),
                torch.softmax(teacher_outputs / temperature, dim=1)
            )
            loss = alpha * cls_loss + (1 - alpha) * distillation_loss

            # loss.backward()
            accelerator.backward(loss)
            optimizer.step()
            train_all_loss += loss.item() * inputs.size(0)
            train_distillation_loss += distillation_loss.item() * inputs.size(0)
            train_cls_loss += cls_loss.item() * inputs.size(0)
            scheduler.step()

        train_y_true = torch.concat(train_y_true).detach().cpu().numpy()
        train_y_pred = torch.concat(train_y_pred).detach().cpu().numpy()
        train_y_score = torch.concat(train_y_score).detach().cpu().numpy()
        session.tb_train_log(
            global_step=epoch,
            metrics={
                'loss': train_all_loss / train_total,
                'distillation_loss': train_distillation_loss / train_total,
                'cls_loss': train_cls_loss / train_total,
                'accuracy': accuracy_score(train_y_true, train_y_pred),
                'mAP': cls_map_score(train_y_true, train_y_score, labels),
                'AUC': cls_auc_score(train_y_true, train_y_score, labels),
                'confusion': plt_export(
                    plt_confusion_matrix,
                    train_y_true, train_y_pred, labels,
                    title=f'Train Confusion Epoch {epoch}',
                ),
            }
        )

        if epoch % eval_epoch == 0:
            model.eval()
            with torch.no_grad():
                test_cls_loss = 0.0
                test_total = 0
                test_y_true, test_y_pred, test_y_score = [], [], []
                for i, (inputs, labels_) in enumerate(tqdm(test_dataloader)):
                    inputs = inputs.float().to(accelerator.device)
                    labels_ = labels_.to(accelerator.device)

                    outputs = model(inputs)
                    test_y_true.append(labels_)
                    test_y_pred.append(outputs.argmax(dim=1))
                    test_y_score.append(torch.softmax(outputs, dim=1))
                    test_total += labels_.shape[0]

                    cls_loss = student_loss_fn(outputs, labels_)
                    test_cls_loss += cls_loss.item() * inputs.size(0)

                test_y_true = torch.concat(test_y_true).cpu().numpy()
                test_y_pred = torch.concat(test_y_pred).cpu().numpy()
                test_y_score = torch.concat(test_y_score).cpu().numpy()
                session.tb_eval_log(
                    global_step=epoch,
                    model=model,
                    metrics={
                        'cls_loss': test_cls_loss / test_total,
                        'accuracy': accuracy_score(test_y_true, test_y_pred),
                        'mAP': cls_map_score(test_y_true, test_y_score, labels),
                        'AUC': cls_auc_score(test_y_true, test_y_score, labels),
                        'confusion': plt_export(
                            plt_confusion_matrix,
                            test_y_true, test_y_pred, labels,
                            title=f'Test Confusion Epoch {epoch}',
                        ),
                        'p_curve': plt_export(
                            plt_p_curve, test_y_true, test_y_score, labels,
                            title=f'Precision Epoch {epoch}',
                        ),
                        'r_curve': plt_export(
                            plt_r_curve, test_y_true, test_y_score, labels,
                            title=f'Recall Epoch {epoch}',
                        ),
                        'pr_curve': plt_export(
                            plt_pr_curve, test_y_true, test_y_score, labels,
                            title=f'PR Curve Epoch {epoch}',
                        ),
                        'f1_curve': plt_export(
                            plt_f1_curve, test_y_true, test_y_score, labels,
                            title=f'F1 Curve Epoch {epoch}',
                        ),
                        'roc_curve': plt_export(
                            plt_roc_curve, test_y_true, test_y_score, labels,
                            title=f'ROC Curve Epoch {epoch}',
                        )
                    }
                )
