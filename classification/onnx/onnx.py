import os
from tempfile import TemporaryDirectory

import onnx
import onnxoptimizer
import onnxsim
import torch
from ditk import logging


def onnx_optimize(model):
    model = onnxoptimizer.optimize(model)
    model, check = onnxsim.simplify(model)
    assert check, "Simplified ONNX model could not be validated"
    return model


def onnx_quick_export(model, example_input, onnx_filename, opset_version: int = 14, verbose: bool = True,
                      no_optimize: bool = False, dynamic_axes=None, no_gpu=False):
    model = model.float()
    if torch.cuda.is_available() and not no_gpu:
        example_input = example_input.cuda()
        model = model.cuda()
    else:
        example_input = example_input.cpu()
        model = model.cpu()

    with torch.no_grad(), TemporaryDirectory() as td:
        onnx_model_file = os.path.join(td, 'model.onnx')
        logging.info(f'Onnx data exporting to {onnx_model_file!r} ...')
        torch.onnx.export(
            model,
            example_input,
            onnx_model_file,
            verbose=verbose,
            input_names=["input"],
            output_names=["output"],

            opset_version=opset_version,
            dynamic_axes=dynamic_axes or {},
        )

        model = onnx.load(onnx_model_file)
        if not no_optimize:
            logging.info('Optimizing onnx model ...')
            model = onnx_optimize(model)

        output_model_dir, _ = os.path.split(onnx_filename)
        if output_model_dir:
            os.makedirs(output_model_dir, exist_ok=True)
        logging.info(f'Complete model saving to {onnx_filename!r} ...')
        onnx.save(model, onnx_filename)
