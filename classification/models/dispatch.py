import timm
import torch


def get_backbone_model(name, labels, **kwargs):
    model = timm.create_model(name, **kwargs)
    if model.num_classes != len(labels):
        model.reset_classifier(num_classes=len(labels))
    model.__arguments__ = {'name': name, 'labels': labels, **kwargs}
    model.__info__ = {}
    return model.float()


def save_model_to_ckpt(model, file):
    arguments = getattr(model, '__arguments__', {}) or {}
    info = getattr(model, '__info__', {}) or {}
    torch.save({
        'state_dict': model.state_dict(),
        'arguments': arguments,
        'info': info,
    }, file)


def load_model_from_ckpt(file):
    data = torch.load(file, map_location='cpu')
    arguments = data['arguments'].copy()
    name = arguments.pop('name')
    labels = arguments.pop('labels')
    model = get_backbone_model(name, labels, **arguments)
    existing_keys = set(model.state_dict())
    state_dict = {key: value for key, value in data['state_dict'].items() if key in existing_keys}
    model.load_state_dict(state_dict)
    model.__info__ = data['info']

    return model
