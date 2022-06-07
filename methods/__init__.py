import torch
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel

from .SWEM import SWEMTrainer, SWEMEvaluator


trainer_map = {
    'SWEM': SWEMTrainer,
}

evaluator_map = {
    'SWEM': SWEMEvaluator,
}


def get_trainer(config, *args, **kwargs):
    assert (config.MODEL.MODEL_NAME in trainer_map.keys())
    return trainer_map[config.MODEL.MODEL_NAME](config, *args, **kwargs)


def get_evaluator(config, *args, **kwargs):
    assert (config.MODEL.MODEL_NAME in evaluator_map.keys())
    return evaluator_map[config.MODEL.MODEL_NAME](config, *args, **kwargs)


def save_model(model, model_path):
    if isinstance(model, DataParallel) or isinstance(model, DistributedDataParallel):
        model = model.module
    torch.save(model.state_dict(), model_path)


def load_model(model, single_obj, model_path, strict=True, cpu=False):
    if cpu:
        loaded_model = torch.load(model_path, map_location='cpu')
    else:
        loaded_model = torch.load(model_path)

    # Maps SO weight (without other_mask) to MO weight (with other_mask)
    for k in list(loaded_model.keys()):
        if k == 'ValueEncoder.conv1.weight':
            if loaded_model[k].shape[1] == 4 and single_obj:
                pads = torch.zeros((64, 1, 7, 7), device=loaded_model[k].device)
                torch.nn.init.orthogonal_(pads)
                loaded_model[k] = torch.cat([loaded_model[k], pads], 1)

    if isinstance(model, DataParallel) or isinstance(model, DistributedDataParallel):
        model.module.load_state_dict(loaded_model, strict=strict)
    else:
        model.load_state_dict(loaded_model, strict=strict)

    del loaded_model
    torch.cuda.empty_cache()


