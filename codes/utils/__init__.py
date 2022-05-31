import shutil

import torch.nn.functional as F

from .logger import *
from .parallel import *
from .visualization import *


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def init_random_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)

    #torch.backends.cudnn.benchmark = True
    #torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# STM
def pad_divide_by(in_img, d, in_size=None):
    if in_size is None:
        h, w = in_img.shape[-2:]
    else:
        h, w = in_size

    if h % d > 0:
        new_h = h + d - h % d
    else:
        new_h = h
    if w % d > 0:
        new_w = w + d - w % d
    else:
        new_w = w
    lh, uh = int((new_h-h) / 2), int(new_h-h) - int((new_h-h) / 2)
    lw, uw = int((new_w-w) / 2), int(new_w-w) - int((new_w-w) / 2)
    pad_array = (int(lw), int(uw), int(lh), int(uh))
    out = F.pad(in_img, pad_array)
    return out, pad_array


def unpad(img, pad):
    if pad[2]+pad[3] > 0:
        img = img[:, :, pad[2]:-pad[3], :]
    if pad[0]+pad[1] > 0:
        img = img[:, :, :, pad[0]:-pad[1]]
    return img


def save_scripts(path, scripts_to_save=None):
    if not os.path.exists(os.path.join(path, 'scripts')):
        os.makedirs(os.path.join(path, 'scripts'))

    if scripts_to_save is not None:
        for script in scripts_to_save:
            dst_path = os.path.join(path, 'scripts', script)
            try:
                shutil.copy(script, dst_path)
            except IOError:
                os.makedirs(os.path.dirname(dst_path))
                shutil.copy(script, dst_path)


def count_model_size(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters()) / 1e6