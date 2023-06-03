import datetime
import os
from functools import partial
from tempfile import TemporaryDirectory
from typing import Optional

import click
from ditk import logging
from huggingface_hub import HfApi, CommitOperationAdd

from .export import export_model_from_workdir
from .utils import GLOBAL_CONTEXT_SETTINGS
from .utils import print_version as _origin_print_version

print_version = partial(_origin_print_version, 'publish')


@click.group(context_settings={**GLOBAL_CONTEXT_SETTINGS})
@click.option('-v', '--version', is_flag=True,
              callback=print_version, expose_value=False, is_eager=True,
              help="Utils publishing models.")
def cli():
    pass  # pragma: no cover


@cli.command('huggingface', context_settings={**GLOBAL_CONTEXT_SETTINGS},
             help='Publish model to huggingface model repository')
@click.option('--workdir', '-w', 'workdir', type=click.Path(file_okay=False, exists=True), required=True,
              help='Work directory of the training.', show_default=True)
@click.option('--imgsize', '-s', 'imgsize', type=int, default=384,
              help='Image size for input.', show_default=True)
@click.option('--non-dynamic', '-D', 'non_dynamic', is_flag=True, type=bool, default=False,
              help='Do not export model with dynamic input height and width.', show_default=True)
@click.option('--verbose', '-V', 'verbose', is_flag=True, type=bool, default=False,
              help='Show verbose information.', show_default=True)
@click.option('--name', '-n', 'name', type=str, default=None,
              help='Name of the checkpoint. Default is the basename of the work directory.', show_default=True)
@click.option('--repository', '-r', 'repository', type=str, required=True,
              help='Repository for publishing model.', show_default=True)
@click.option('--revision', '-R', 'revision', type=str, default='main',
              help='Revision for pushing the model.', show_default=True)
def cli(workdir: str, imgsize: int, non_dynamic: bool, verbose: bool, name: Optional[str],
        repository: str, revision: str):
    logging.try_init_root(logging.INFO)

    hf_client = HfApi(token=os.environ['HF_ACCESS_TOKEN'])
    logging.info(f'Initialize repository {repository!r}')
    hf_client.create_repo(repo_id=repository, repo_type='model', exist_ok=True)

    with TemporaryDirectory() as td:
        name = name or os.path.basename(os.path.abspath(workdir))
        files = export_model_from_workdir(workdir, td, imgsize, non_dynamic, verbose, name)
        current_time = datetime.datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')
        commit_message = f"Publish model {name}, on {current_time}"
        logging.info(f'Publishing model {name!r} to repository {repository!r} ...')
        hf_client.create_commit(
            repository,
            [
                CommitOperationAdd(
                    path_in_repo=f'{name}/{filename}',
                    path_or_fileobj=local_file,
                ) for local_file, filename in files
            ],
            commit_message=commit_message,
            repo_type='model',
            revision=revision,
        )


if __name__ == '__main__':
    cli()
