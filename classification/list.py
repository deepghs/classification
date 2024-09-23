import io
import json
import os
import re
import warnings
from functools import partial
from typing import Tuple
from urllib.parse import quote

import click
import numpy as np
import pandas as pd
import torch
from ditk import logging
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory
from hfutils.operate import upload_directory_as_directory
from hfutils.utils import hf_fs_path
from huggingface_hub import HfFileSystem, hf_hub_url, HfApi
from huggingface_hub.constants import ENDPOINT
from huggingface_hub.hf_api import RepoFile
from tqdm.auto import tqdm

from classification.models import load_model_from_ckpt
from classification.train import torch_model_profile
from .utils import GLOBAL_CONTEXT_SETTINGS, markdown_to_df
from .utils import print_version as _origin_print_version

print_version = partial(_origin_print_version, 'list')


@click.group(context_settings={**GLOBAL_CONTEXT_SETTINGS})
@click.option('-v', '--version', is_flag=True,
              callback=print_version, expose_value=False, is_eager=True,
              help="List published models.")
def cli():
    pass  # pragma: no cover


def _name_process(name: str):
    words = re.split(r'[\W_]+', name)
    return ' '.join([
        word.capitalize() if re.fullmatch('^[a-z0-9]+$', word) else word
        for word in words
    ])


_PERCENTAGE_METRICS = ('accuracy',)
HUGGINGFACE_CO_PAGE_TEMPLATE = ENDPOINT + "/{repo_id}/blob/{revision}/{filename}"


@cli.command('huggingface', context_settings={**GLOBAL_CONTEXT_SETTINGS},
             help='Publish model to huggingface model repository')
@click.option('--repository', '-r', 'repository', type=str, required=True,
              help='Repository for publishing model.', show_default=True)
@click.option('--revision', '-R', 'revision', type=str, default='main',
              help='Revision for pushing the model.', show_default=True)
@click.option('--column', '-c', 'columns', type=str, multiple=True,
              help='Columns to be listed.')
@click.option('--image', '-I', 'show_image', is_flag=True, type=bool, default=False,
              help='Show image in table.', show_default=True)
def huggingface(repository: str, revision: str, columns: Tuple[str, ...], show_image: bool):
    logging.try_init_root(logging.INFO)
    columns = columns or ('name', 'FLOPS', 'params', 'accuracy', 'AUC', 'confusion', 'labels')

    hf_fs = HfFileSystem(token=os.environ.get('HF_TOKEN'))
    hf_client = HfApi(token=os.environ.get('HF_TOKEN'))

    names = [fn.split('/')[-2] for fn in hf_fs.glob(f'{repository}@{revision}/*/model.ckpt')]
    logging.info(f'{plural_word(len(names), "model")} detected in {repository}@{revision}')

    rows = []
    for name in tqdm(names):
        files = hf_fs.ls(f'{repository}@{revision}/{name}')
        meta_info = json.loads(hf_fs.read_text(f'{repository}@{revision}/{name}/meta.json'))
        imgsize = meta_info.get('img_size', 384)
        values = {'name': name}

        model = load_model_from_ckpt(hf_client.hf_hub_download(repository, f'{name}/model.ckpt', revision=revision))
        input_ = torch.randn(1, 3, imgsize, imgsize)
        flops, params = torch_model_profile(model, input_)
        values['FLOPS'] = f'{flops / 1e9:.2f}G'
        values['params'] = f'{params / 1e6:.2f}M'

        repo_file: RepoFile = list(hf_client.get_paths_info(
            repo_id=repository,
            repo_type='model',
            paths=[f'{name}/model.ckpt'],
            expand=True,
        ))[0]
        last_commit_at = repo_file.last_commit.date.timestamp()

        with open(hf_client.hf_hub_download(repository, f'{name}/meta.json', revision=revision), 'r') as f:
            values['labels'] = ', '.join([f'`{x}`' for x in json.load(f)['labels']])

        plots = {}
        for file in files:
            filename = os.path.basename(file['name'])
            if filename.startswith('plot_'):
                key = os.path.splitext(filename)[0][5:]
                if show_image:
                    value = hf_hub_url(repository, f'{name}/{filename}', revision=revision)
                else:
                    value = HUGGINGFACE_CO_PAGE_TEMPLATE.format(
                        repo_id=repository,
                        revision=quote(revision, safe=""),
                        filename=quote(f'{name}/{filename}'),
                    )
                plots[key] = value

        with open(hf_client.hf_hub_download(repository, f'{name}/metrics.json', revision=revision), 'r') as f:
            metrics = json.load(f)

        item = {}
        for c in columns:
            key = _name_process(c)
            if c in values:
                item[key] = values[c]
            elif c in plots:
                if show_image:
                    item[key] = f'![{c}]({plots[c]})'
                else:
                    item[key] = f'[{c}]({plots[c]})'
            elif c in metrics:
                if c in _PERCENTAGE_METRICS:
                    item[key] = f'{metrics[c] * 100.0:.2f}%'
                elif isinstance(metrics[c], float):
                    item[key] = f'{metrics[c]:.4f}'
                else:
                    item[key] = metrics[c]
            else:
                warnings.warn(f'Unknown column {c!r} for model {name!r}.')
                item[key] = 'N/A'

        item['created_at'] = last_commit_at
        rows.append(item)

    df = pd.DataFrame(rows)
    df = df.sort_values(by=['created_at'], ascending=[False])
    del df['created_at']
    df = df.replace(np.nan, 'N/A')

    with TemporaryDirectory() as td:
        with open(os.path.join(td, 'README.md'), 'w') as f:
            if not hf_fs.exists(hf_fs_path(
                    repo_id=repository,
                    repo_type='model',
                    filename='README.md',
                    revision=revision,
            )):
                print(df.to_markdown(index=False, numalign="center", stralign="center"), file=f)

            else:
                table_printed = False
                tb_lines = []
                with io.StringIO(hf_fs.read_text(hf_fs_path(
                        repo_id=repository,
                        repo_type='model',
                        filename='README.md',
                        revision=revision,
                )).rstrip() + os.linesep * 2) as ifx:
                    for line in ifx:
                        line = line.rstrip()
                        if line.startswith('|') and not table_printed:
                            tb_lines.append(line)
                        else:
                            if tb_lines:
                                df_c = markdown_to_df(os.linesep.join(tb_lines))
                                if 'Name' in df_c.columns and 'FLOPS' in df_c.columns and \
                                        'Params' in df_c.columns and 'Labels' in df_c.columns:
                                    print(df.to_markdown(index=False, numalign="center", stralign="center"), file=f)
                                    table_printed = True
                                    tb_lines.clear()
                                else:
                                    print(os.linesep.join(tb_lines), file=f)
                            print(line, file=f)

                if not table_printed:
                    print(df.to_markdown(index=False, numalign="center", stralign="center"), file=f)

        upload_directory_as_directory(
            repo_id=repository,
            repo_type='model',
            revision=revision,
            path_in_repo='.',
            local_directory=td,
            message=f'Sync README for {repository}',
            hf_token=os.environ.get('HF_TOKEN'),
        )


if __name__ == '__main__':
    cli()
