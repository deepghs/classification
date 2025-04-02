import json
import os
import warnings
from typing import Optional, List

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
from imgutils.preprocess import parse_torchvision_transforms
from .base import register_task_type, DEFAULT_TASK, put_meta_at_workdir
from .metrics import cls_map_score, cls_auc_score
from .profile import torch_model_profile
from .session import _load_last_ckpt, TrainSession
from ..losses import get_loss_fn
from ..models import get_backbone_model
from ..plot import plt_export, plt_confusion_matrix, plt_pr_curve, plt_p_curve, plt_r_curve, plt_f1_curve, \
    plt_roc_curve, plot_samples
from ..dataset import TestDatasetWrapper, TestDatasetVisualWrapper

_PRESET_KWARGS = {
    'pretrained': True,
}

_TASK_NAME = 'classify'
assert _TASK_NAME == DEFAULT_TASK, f'Task name should be the same as DEFAULT_TASK, but {_TASK_NAME!r} found.'

register_task_type(
    _TASK_NAME,
    onnx_conf=dict(
        use_softmax=True,
    ),
)


def train_simple(
        workdir: str, model_name: str, labels: List[str],
        train_dataset: Dataset, test_dataset: Dataset,
        batch_size: int = 16, max_epochs: int = 500, learning_rate: float = 0.001,
        weight_decay: float = 1e-3, num_workers: Optional[int] = 8, eval_epoch: int = 5,
        key_metric: Literal['accuracy', 'AUC', 'mAP'] = 'accuracy', loss='focal',
        loss_weight=None, seed: Optional[int] = 0, loss_args: Optional[dict] = None,
        img_size: int = 384, preprocessor=None, **model_args):
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

    model_args = {**_PRESET_KWARGS, 'img_size': img_size, **model_args}
    model, _model_name, _labels, _model_args = _load_last_ckpt(workdir)
    if model:
        logging.info(f'Load previous ckpt from work directory {workdir!r}.')
        assert _model_name == model_name, \
            f'Resumed model name {_model_args!r} not match with specified name {model_name!r}.'
        assert _labels == labels, \
            f'Resumed labels {_labels!r} not match with specified labels {labels!r}.'
        if _model_args != model_args:
            warnings.warn(f'Resumed model arguments {_model_args!r} '
                          f'not match with specified arguments {model_args!r}.')
        previous_epoch = model.__info__['step']
        logging.info(f'Previous step is {previous_epoch!r}')
    else:
        logging.info(f'No previous ckpt found, load new model {model_name!r} with {model_args!r}.')
        model = get_backbone_model(model_name, labels, **model_args)
        previous_epoch = 0

    sample_input, *_, _ = test_dataset[0]
    torch_model_profile(model, sample_input.unsqueeze(0))  # profile the model

    num_workers = num_workers or min(os.cpu_count(), batch_size)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, drop_last=True
    )
    test_dataloader = DataLoader(TestDatasetWrapper(test_dataset), batch_size=batch_size, num_workers=num_workers)
    visual_dataset = TestDatasetVisualWrapper(test_dataset)

    if loss_weight is None:
        loss_weight = torch.ones(len(labels), dtype=torch.float)
    loss_fn = get_loss_fn(loss, len(labels), loss_weight, **dict(loss_args or {}), reduction='mean')
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = lr_scheduler.OneCycleLR(
        optimizer, max_lr=learning_rate,
        steps_per_epoch=len(train_dataloader), epochs=max_epochs,
        pct_start=0.15, final_div_factor=20.
    )

    # noinspection PyTypeChecker
    model, optimizer, train_dataloader, test_dataloader, scheduler, loss_fn = \
        accelerator.prepare(model, optimizer, train_dataloader, test_dataloader, scheduler, loss_fn)
    cm_size = max(6.0, len(labels) * 0.9)

    if preprocessor is not None:
        logging.info('Saving the preprocessor ...')
        with open(os.path.join(workdir, 'preprocess.json'), 'w') as f:
            json.dump({
                'stages': parse_torchvision_transforms(preprocessor)
            }, f, indent=4, sort_keys=True, ensure_ascii=False)

    logging.info(f'Model\'s arguments: {model.__arguments__!r}, info: {model.__info__!r}.')
    session = TrainSession(workdir, key_metric=key_metric)
    logging.info('Training start!')
    for epoch in range(previous_epoch + 1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        train_total = 0
        train_y_true, train_y_pred, train_y_score = [], [], []
        for i, (inputs, labels_) in enumerate(tqdm(train_dataloader)):
            inputs = inputs.float()
            inputs = inputs.to(accelerator.device)
            labels_ = labels_.to(accelerator.device)

            optimizer.zero_grad()
            outputs = model(inputs)
            train_y_true.append(labels_)
            train_y_pred.append(outputs.argmax(dim=1))
            train_y_score.append(torch.softmax(outputs, dim=1))
            train_total += labels_.shape[0]

            loss = loss_fn(outputs, labels_)
            # loss.backward()
            accelerator.backward(loss)
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            scheduler.step()

        train_y_true = torch.concat(train_y_true).detach().cpu().numpy()
        train_y_pred = torch.concat(train_y_pred).detach().cpu().numpy()
        train_y_score = torch.concat(train_y_score).detach().cpu().numpy()
        session.tb_train_log(
            global_step=epoch,
            metrics={
                'loss': train_loss / train_total,
                'accuracy': accuracy_score(train_y_true, train_y_pred),
                'mAP': cls_map_score(train_y_true, train_y_score, labels),
                'AUC': cls_auc_score(train_y_true, train_y_score, labels),
                'confusion': plt_export(
                    plt_confusion_matrix,
                    train_y_true, train_y_pred, labels,
                    title=f'Train Confusion Epoch {epoch}',
                    figsize=(cm_size, cm_size),
                ),
            }
        )

        if epoch % eval_epoch == 0:
            model.eval()
            with torch.no_grad():
                test_loss = 0.0
                test_total = 0
                test_y_true, test_y_pred, test_y_score = [], [], []
                test_need_visual, test_ids = True, []
                for i, (ids, inputs, *_visuals, labels_) in enumerate(tqdm(test_dataloader)):
                    inputs = inputs.float().to(accelerator.device)
                    labels_ = labels_.to(accelerator.device)

                    outputs = model(inputs)
                    test_y_true.append(labels_)
                    test_y_pred.append(outputs.argmax(dim=1))
                    test_y_score.append(torch.softmax(outputs, dim=1))
                    test_total += labels_.shape[0]

                    loss = loss_fn(outputs, labels_)
                    test_loss += loss.item() * inputs.size(0)

                    if not _visuals:
                        test_need_visual = False
                    test_ids.append(ids.cpu())

                test_y_true = torch.concat(test_y_true).cpu().numpy()
                test_y_pred = torch.concat(test_y_pred).cpu().numpy()
                test_y_score = torch.concat(test_y_score).cpu().numpy()
                test_ids = torch.concat(test_ids).numpy()
                if test_need_visual:
                    logging.info('Creating visual samples ...')
                    test_plot_visuals = {
                        f'sample_{labels[li]}': plot_samples(
                            test_y_true, test_y_pred,
                            test_ids, visual_dataset,
                            concen_cls=li, labels=labels
                        )
                        for li in range(len(labels))
                    }
                else:
                    test_plot_visuals = {}

                session.tb_eval_log(
                    global_step=epoch,
                    model=model,
                    metrics={
                        'loss': test_loss / test_total,
                        'accuracy': accuracy_score(test_y_true, test_y_pred),
                        'mAP': cls_map_score(test_y_true, test_y_score, labels),
                        'AUC': cls_auc_score(test_y_true, test_y_score, labels),
                        'confusion': plt_export(
                            plt_confusion_matrix,
                            test_y_true, test_y_pred, labels,
                            title=f'Test Confusion Epoch {epoch}',
                            figsize=(cm_size, cm_size),
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
                        ),
                        **test_plot_visuals,
                    }
                )
