import os
import torch
import argparse

import torch.distributed as dist
import torch.multiprocessing as mp

from methods import get_trainer
from configs.config import VOSConfig
from utils import init_random_seed
# ignore warnings
import warnings
warnings.filterwarnings('ignore')


def main():
    trainer = get_trainer(config, name=args.backend, is_dist=is_dist, rank=rank)
    # training
    trainer.train()


def env_setup():
    #os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '18888'
    rank = 0
    num_gpu = 1
    is_dist = False
    if 'WORLD_SIZE' in os.environ:
        num_gpu = int(os.environ['WORLD_SIZE'])
        is_dist = num_gpu > 1

    if is_dist:
        rank = args.local_rank
        if mp.get_start_method(allow_none=True) != 'spawn':
            mp.set_start_method('spawn', force=True)
        torch.cuda.set_device(rank)
        dist.init_process_group(backend='nccl', init_method='env://')
        dist.barrier()
    init_random_seed(config.DATASET.SEED + rank)
    #torch.autograd.set_detect_anomaly(True)
    return rank, is_dist, num_gpu


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch VOS Training')
    parser.add_argument('--opt_level', dest='opt_level', type=str, default='O1')  # [O0|O1|O2|O3]
    parser.add_argument("--local_rank", dest='local_rank', default=0, type=int)
    parser.add_argument('--amp', dest='AMP', action='store_true',
                        help='Use AMP during training or not.')

    # model parameters
    parser.add_argument("--model", dest='MODEL_NAME', default='SMEM', type=str,
                        help='Model name.')
    parser.add_argument("--backbone", dest='BACKBONE', default='resnet50', type=str,
                        help='The backbone for key encoder.')
    parser.add_argument("--key_dim", dest='KEYDIM', default=128, type=int,
                        help='The number of key dimension [64|128].')
    parser.add_argument('--resume', dest='resume', type=str, default=None,
                        help='The path to the checkpoint (default: none).')
    parser.add_argument('--from_scratch', dest='from_scratch', action='store_true',
                        help='Training from start or resume iterations.')

    # exp parameters
    parser.add_argument("--stage", dest='STAGE', default=0, type=int,
                        help='Training stage [0:Image|1:DAVIS|2:YTVOS19|3:DAVIS+YTVOS19].')
    parser.add_argument("--stage_name", dest='STAGE_NAME', default='S0', type=str,
                        help='Training stage [S0|S1|S2|S3].')
    parser.add_argument("--num_obj", dest='MAX_NUM_OBJS', default=2, type=int,
                        help='The number of maximum number of objects during training [>=1].')
    parser.add_argument("--batch_size", dest='batch_size', default=8, type=int,
                        help='The number of batch_size.')
    parser.add_argument("--lr", dest='BASE_LR', default=2e-5, type=float,
                        help='Learning rate.')
    parser.add_argument('--backend', dest='backend', type=str, default='baseline',
                        help='The name of exp.')

    # matching parameters
    # EM hyper-parameters if needed
    parser.add_argument('--em_iter', dest='NUM_EM_ITERS', default=4, type=int,
                        help='The number of iterations in EM')
    parser.add_argument("--num_bases", dest='NUM_BASES', default=128, type=int,
                        help='The number of bases [32|64|128|256|512].')
    parser.add_argument("--top_l", dest='TOPL', default=64, type=int,
                        help='Top-l matching scores, <= number of bases.')
    parser.add_argument('--tau', dest='EM_TAU', default=0.05, type=float,
                        help='The tau in EM.')

    args = parser.parse_args()
    config = VOSConfig(args)

    rank, is_dist, num_gpu = env_setup()
    config.DATALOADER.IMG_PER_GPU = args.batch_size // num_gpu
    config.DATALOADER.NUM_WORKERS = min(config.DATALOADER.IMG_PER_GPU * 2, 16)

    if args.resume is not None:
        config.RESUME = os.path.join(config.CODE_ROOT, 'logs', args.MODEL_NAME, args.resume)
        config.FROM_SCRATCH = args.from_scratch

    if rank <= 0:
        print(f'Training {config.MODEL.MODEL_NAME} with batch size {args.batch_size} on {num_gpu} GPUs.')
        for key, value in config.__dict__.items():
            print(f'{key}: {value}')
        for key, value in config.DATASET.items():
            if not isinstance(value, dict):
                print(f'{key}: {value}')
        for key, value in config.DATALOADER.items():
            print(f'{key}: {value}')
        for key, value in config.MODEL.items():
            print(f'{key}: {value}')
        for key, value in config.SOLVER.items():
            print(f'{key}: {value}')

    main()

    if is_dist:
        dist.destroy_process_group()


