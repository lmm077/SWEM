import os
import numpy as np
import random
import torch
import torch.distributed as dist


def is_dist():
    return dist.is_initialized()


def get_rank():
    if not is_dist():
        return 0
    else:
        return dist.get_rank()


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= int(os.environ['WORLD_SIZE'])
    return rt


def init_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)

    #torch.backends.cudnn.benchmark = True
    #torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)




