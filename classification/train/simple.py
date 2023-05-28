import os
import warnings
from typing import Optional, List

import torch
from accelerate import Accelerator
from hbutils.random import global_seed
from sklearn.metrics import accuracy_score
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from .metrics import cls_map_score
from .session import _load_last_ckpt, TrainSession
from ..losses import FocalLoss
from ..models import get_backbone_model
from ..plot import plt_export, plt_confusion_matrix, plt_pr_curve, plt_p_curve, plt_r_curve, plt_f1_curve


def train_simple(
        workdir: str, model_name: str, labels: List[str],
        train_dataset: Dataset, test_dataset: Dataset,
        batch_size: int = 16, max_epochs: int = 500, learning_rate: float = 0.001,
        weight_decay: float = 1e-3, num_workers: Optional[int] = 8, eval_epoch: int = 5,
        loss_weight=None, seed: Optional[int] = 0, **model_args):
    if seed is not None:
        # native random, numpy, torch and faker's seeds are includes
        # if you need to register more library for seeding, see:
        # https://hansbug.github.io/hbutils/main/api_doc/random/state.html#register-random-source
        global_seed(seed)

    os.makedirs(workdir, exist_ok=True)
    accelerator = Accelerator(
        # mixed_precision=self.cfgs.mixed_precision,
        step_scheduler_with_optimizer=False,
    )

    model, _model_name, _labels, _model_args = _load_last_ckpt(workdir)
    if model:
        assert _model_name == model_name, \
            f'Resumed model name {_model_args!r} not match with specified name {model_name!r}.'
        assert _labels == labels, \
            f'Resumed labels {_labels!r} not match with specified labels {labels!r}.'
        if _model_args != model_args:
            warnings.warn(f'Resumed model arguments {_model_args!r} '
                          f'not match with specified arguments {model_args!r}.')
        previous_epoch = model.__info__['step']
    else:
        model = get_backbone_model(model_name, labels, **model_args)
        previous_epoch = 0

    num_workers = num_workers or min(os.cpu_count(), batch_size)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, drop_last=True
    )
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    if loss_weight is None:
        loss_weight = torch.ones(len(labels), dtype=torch.float)
    loss_fn = FocalLoss(weight=loss_weight).to(accelerator.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = lr_scheduler.OneCycleLR(
        optimizer, max_lr=learning_rate,
        steps_per_epoch=len(train_dataloader), epochs=max_epochs,
        pct_start=0.15, final_div_factor=20.
    )

    # noinspection PyTypeChecker
    model, optimizer, train_dataloader, test_dataloader, scheduler = \
        accelerator.prepare(model, optimizer, train_dataloader, test_dataloader, scheduler)

    session = TrainSession(workdir)
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
        session.tb_train_log(
            global_step=epoch,
            metrics={
                'loss': train_loss / train_total,
                'accuracy': accuracy_score(train_y_true, train_y_pred),
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
                test_loss = 0.0
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

                    loss = loss_fn(outputs, labels_)
                    test_loss += loss.item() * inputs.size(0)

                test_y_true = torch.concat(test_y_true).cpu().numpy()
                test_y_pred = torch.concat(test_y_pred).cpu().numpy()
                test_y_score = torch.concat(test_y_score).cpu().numpy()
                session.tb_eval_log(
                    global_step=epoch,
                    model=model,
                    metrics={
                        'loss': test_loss / test_total,
                        'accuracy': accuracy_score(test_y_true, test_y_pred),
                        'mAP': cls_map_score(test_y_true, test_y_score, labels),
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
                    }
                )
