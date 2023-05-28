import os.path
from functools import partial

import click
from ditk import logging

from .export import export_onnx_from_ckpt, validate_onnx_model
from ..utils import GLOBAL_CONTEXT_SETTINGS
from ..utils import print_version as _origin_print_version

print_version = partial(_origin_print_version, 'onnx')


@click.group(context_settings={**GLOBAL_CONTEXT_SETTINGS})
@click.option('-v', '--version', is_flag=True,
              callback=print_version, expose_value=False, is_eager=True,
              help="Utils with onnx models.")
def cli():
    pass  # pragma: no cover


@cli.command('export', help='Export existing checkpoint to onnx')
@click.option('--input', '-i', 'model_filename', type=click.Path(dir_okay=False, exists=True), required=True,
              help='Input checkpoint to export.', show_default=True)
@click.option('--imgsize', '-s', 'imgsize', type=int, default=384,
              help='Image size for input.', show_default=True)
@click.option('--non-dynamic', '-D', 'non_dynamic', is_flag=True, type=bool, default=False,
              help='Do not export model with dynamic input height and width.', show_default=True)
@click.option('--verbose', '-V', 'verbose', is_flag=True, type=bool, default=False,
              help='Show verbose information.', show_default=True)
@click.option('--output', '-o', 'onnx_filename', type=click.Path(dir_okay=False), default=None,
              help='Output file of onnx model', show_default=True)
def export(model_filename: str, imgsize: int, non_dynamic: bool, verbose: bool, onnx_filename):
    logging.try_init_root(logging.INFO)
    export_onnx_from_ckpt(model_filename, onnx_filename, 14, verbose, imgsize, not non_dynamic)
    validate_onnx_model(onnx_filename)


@cli.command('dump', help='Dump onnx model from existing work directory.')
@click.option('--workdir', '-w', 'workdir', type=click.Path(file_okay=False, exists=True), required=True,
              help='Work directory of the training.', show_default=True)
@click.option('--imgsize', '-s', 'imgsize', type=int, default=384,
              help='Image size for input.', show_default=True)
@click.option('--non-dynamic', '-D', 'non_dynamic', is_flag=True, type=bool, default=False,
              help='Do not export model with dynamic input height and width.', show_default=True)
@click.option('--verbose', '-V', 'verbose', is_flag=True, type=bool, default=False,
              help='Show verbose information.', show_default=True)
def dump(workdir: str, imgsize: int, non_dynamic: bool, verbose: bool):
    logging.try_init_root(logging.INFO)
    model_filename = os.path.join(workdir, 'ckpts', 'best.ckpt')
    onnx_filename = os.path.join(workdir, 'onnxs', 'best.onnx')
    export_onnx_from_ckpt(model_filename, onnx_filename, 14, verbose, imgsize, not non_dynamic)
    validate_onnx_model(onnx_filename)


if __name__ == '__main__':
    cli()
