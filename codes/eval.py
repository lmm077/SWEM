
import argparse
import torch

from configs.config import VOSConfig
from methods import get_evaluator

torch.set_grad_enabled(False)


def get_args():
    parser = argparse.ArgumentParser(description='Eval VOSNet')

    # model parameters
    parser.add_argument("--model", dest='MODEL_NAME', default='MEM', type=str,
                        help='Model name [SWEM|STCN|STM|EM|WEM].')
    parser.add_argument("--backbone", dest='BACKBONE', default='resnet50', type=str,
                        help='The backbone for key encoder [resnet50|resnet18].')
    parser.add_argument("--key_dim", dest='KEYDIM', default=128, type=int,
                        help='The number of key dimension [64|128].')
    parser.add_argument('--resume', dest='RESUME', type=str, default=None,
                        help='The path to the checkpoint (default: none).')

    # eval parameters
    parser.add_argument("--stage", dest='STAGE', default=0, type=int,
                        help='Training stage [0:Image|1:DAVIS|2:YTVOS18|3:YTVOS19|4:DAVIS+YTVOS18|5:DAVIS+YTVOS19].')
    parser.add_argument("--stage_name", dest='STAGE_NAME', default='S0', type=str,
                        help='Training stage [S0|S1|S2|S3|S4|S5|S01|S02|S03|S04|S05].')
    parser.add_argument("--num_obj", dest='MAX_NUM_OBJS', default=2, type=int,
                        help='The number of maximum number of objects during training [>1].')
    parser.add_argument('--backend', dest='backend', type=str, default='baseline',
                        help='The name of exp.')
    parser.add_argument('--eval_set', dest='eval_set', type=str, default='DAVIS16', required=True,
                        help='[DAVIS16|DAVIS17|YTVOS18|YTVOS19]')
    parser.add_argument('--ssize', dest='ssize',
                        default=480, type=int,
                        help='the shorter size of the input image')

    # matching parameters
    # EM hyper-parameters if needed
    parser.add_argument('--em_iter', dest='NUM_EM_ITERS', default=4, type=int,
                        help='The number of iterations in EM')
    parser.add_argument("--num_bases", dest='NUM_BASES', default=128, type=int,
                        help='The number of bases [32|64|128|256|512].')
    parser.add_argument("--top_l", dest='TOPL', default=64, type=int,
                        help='Top-l matching scores, <= number of bases.')
    parser.add_argument('--tau', dest='EM_TAU', default=0.05, type=float,
                        help='The tau in WEM.')

    return parser.parse_args()


def main():
    # construction model
    evaluator = get_evaluator(config, name=args.backend, eval_set=args.eval_set, rsize=args.ssize, clip_len=32)
    evaluator.val()


if __name__ == '__main__':
    args = get_args()
    config = VOSConfig(args)
    main()

