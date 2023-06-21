import torch
from ditk import logging


def torch_model_profile(model, input_):
    from thop import profile

    with torch.no_grad():
        flops, params = profile(model, (input_,))

    logging.info(f'Params: {params / 1e6:.2f}M, FLOPs: {flops / 1e9:.2f}G.')

    return flops, params
