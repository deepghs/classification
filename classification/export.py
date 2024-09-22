import glob
import json
import os
import re
import shutil
import zipfile
from functools import partial
from typing import Optional, List, Tuple

import click
import numpy as np
import torch
from PIL import Image
from ditk import logging
from hbutils.encoding import sha3

from .models import load_model_from_ckpt
from .onnx.export import export_onnx_from_ckpt, validate_onnx_model
from .train import get_export_config, get_task_type_from_workdir
from .utils import GLOBAL_CONTEXT_SETTINGS
from .utils import print_version as _origin_print_version

_LOG_FILE_PATTERN = re.compile(r'^events\.out\.tfevents\.(?P<timestamp>\d+)\.(?P<machine>[^.]+)\.(?P<extra>[\s\S]+)$')


def export_model_from_workdir(workdir, export_dir, imgsize: int = None, non_dynamic: bool = True,
                              verbose: bool = False, name: Optional[str] = None,
                              logfile_anonymous: bool = True) -> List[Tuple[str, str]]:
    task = get_task_type_from_workdir(workdir)
    export_config = get_export_config(task)
    model_filename = os.path.join(workdir, 'ckpts', 'best.ckpt')
    name = name or os.path.basename(os.path.abspath(workdir))

    model = load_model_from_ckpt(model_filename)
    _info = model.__info__

    metrics = {}
    plots = {}
    for key, value in _info.items():
        if isinstance(value, (int, float, str, type(None))) or \
                (isinstance(value, (torch.Tensor, np.ndarray)) and not value.shape):
            if isinstance(value, (torch.Tensor, np.ndarray)):
                value = value.tolist()

            metrics[key] = value
        elif isinstance(value, Image.Image):
            plots[key] = value
        else:
            logging.warn(f'Argument {key!r} is a {type(value)}, unable to export.')

    os.makedirs(export_dir, exist_ok=True)
    files = []

    ckpt_file = os.path.join(export_dir, f'{name}.ckpt')
    logging.info(f'Copying checkpoint to {ckpt_file!r}')
    shutil.copyfile(model_filename, ckpt_file)
    files.append((ckpt_file, 'model.ckpt'))

    meta_file = os.path.join(export_dir, f'{name}_meta.json')
    logging.info(f'Exporting meta-information of model to {meta_file!r}')
    with open(meta_file, 'w') as f:
        json.dump(model.__arguments__, f, sort_keys=True, indent=4, ensure_ascii=False)
    files.append((meta_file, 'meta.json'))

    metrics_file = os.path.join(export_dir, f'{name}_metrics.json')
    logging.info(f'Recording metrics to {metrics_file!r}')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, sort_keys=True, indent=4, ensure_ascii=False)
    files.append((metrics_file, 'metrics.json'))

    for key, value in plots.items():
        plt_file = os.path.join(export_dir, f'{name}_plot_{key}.png')
        logging.info(f'Save plot figure {key} to {plt_file!r}')
        value.save(plt_file)
        files.append((plt_file, f'plot_{key}.png'))

    onnx_file = os.path.join(export_dir, f'{name}.onnx')
    export_onnx_from_ckpt(model_filename, onnx_file, 14, verbose, imgsize, not non_dynamic, **export_config['onnx'])
    validate_onnx_model(onnx_file)
    files.append((onnx_file, 'model.onnx'))

    for logfile in glob.glob(os.path.join(workdir, 'events.out.tfevents.*')):
        logging.info(f'Tensorboard file {logfile!r} found.')
        matching = _LOG_FILE_PATTERN.fullmatch(os.path.basename(logfile))
        assert matching, f'Log file {logfile!r}\'s name not match with pattern {_LOG_FILE_PATTERN.pattern}.'

        timestamp = matching.group('timestamp')
        machine = matching.group('machine')
        if logfile_anonymous:
            machine = sha3(machine.encode(), n=224)
        extra = matching.group('extra')

        final_name = f'events.out.tfevents.{timestamp}.{machine}.{extra}'
        files.append((logfile, final_name))

    return files


print_version = partial(_origin_print_version, 'onnx')


@click.command(context_settings={**GLOBAL_CONTEXT_SETTINGS})
@click.option('-v', '--version', is_flag=True,
              callback=print_version, expose_value=False, is_eager=True,
              help="Utils with exporting best checkpoints.")
@click.option('--workdir', '-w', 'workdir', type=click.Path(file_okay=False, exists=True), required=True,
              help='Work directory of the training.', show_default=True)
@click.option('--imgsize', '-s', 'imgsize', type=int, default=None,
              help='Image size for input.', show_default=True)
@click.option('--verbose', '-V', 'verbose', is_flag=True, type=bool, default=False,
              help='Show verbose information.', show_default=True)
@click.option('--name', '-n', 'name', type=str, default=None,
              help='Name of the checkpoint. Default is the basename of the work directory.', show_default=True)
def cli(workdir: str, imgsize: int, verbose: bool, name: Optional[str]):
    logging.try_init_root(logging.INFO)
    export_dir = os.path.join(workdir, 'export')
    files = export_model_from_workdir(
        workdir=workdir,
        export_dir=export_dir,
        imgsize=imgsize,
        non_dynamic=True,
        verbose=verbose,
        name=name,
        logfile_anonymous=True,
    )

    zip_file = os.path.join(export_dir, f'{name}.zip')
    logging.info(f'Packing all the above file to archive {zip_file!r}')
    with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file, inner_name in files:
            zf.write(file, inner_name)


if __name__ == '__main__':
    cli()
