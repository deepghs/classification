import json
import os.path
from typing import Mapping, Any

DEFAULT_TASK = 'classify'

_KNOWN_TASKS_CONFIG = {}


def register_task_type(task, onnx_conf: Mapping[str, Any]):
    if task in _KNOWN_TASKS_CONFIG:
        raise KeyError(f'Task already exist - {task!r}.')

    _KNOWN_TASKS_CONFIG[task] = {
        'onnx': onnx_conf,
    }


def get_export_config(task):
    if task not in _KNOWN_TASKS_CONFIG:
        raise KeyError(f'Task type not found - {task!r}.')

    return _KNOWN_TASKS_CONFIG[task]


_META_FILE = 'meta.json'


def put_meta_at_workdir(workdir: str, task: str):
    with open(os.path.join(workdir, _META_FILE), 'w') as f:
        json.dump({
            'task': task,
        }, f, ensure_ascii=False, indent=4, sort_keys=True)


def get_task_type_from_workdir(workdir: str) -> str:
    meta_file = os.path.join(workdir, _META_FILE)
    if os.path.exists(meta_file):
        with open(meta_file, 'r') as f:
            return json.load(f)['task']
    else:
        return DEFAULT_TASK
