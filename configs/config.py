# modified with https://github.com/lyxok1/STM-Training/blob/master/options.py
from easydict import EasyDict as edict


class VOSConfig:
    # ------------------------------------------ dataset ---------------------------------------------
    DATASET = edict()
    DATASET.PHASE = 'train'
    DATASET.SEED = 123
    DATASET.NUM_SAMPLE_PER_SEQ = 3                   # number of sampled frames during training
    DATASET.MAX_NUM_OBJS = 2                         # the maximum number of sampled objects during training
    DATASET.ONLY_VALID = True                        # only calculate loss for valid objects (ignore empty objects)
    DATASET.PRETRAIN_SET = ['BIG_small', 'HRSOD_small', 'FSS', 'ECSSD', 'DUTS']
    DATASET.PRETRAIN_SET_RATIO = [5, 5, 1, 1, 1]
    DATASET.MAINTRAIN_SET = ['DAVIS17', 'YTVOS19']
    DATASET.DATA_FREQ = [1, 1]
    DATASET.PATH = '/home/linzhihui/labs/HumanFace/Video/VOS/codes/datasets'

    DATASET.VID_LOAD_SIZE = 480                      # (h, w) = (480, 864)
    DATASET.VID_CROP_SIZE = (384, 384)

    DATASET.INFO = edict()
    DATASET.INFO.PRETRAIN = {
        'root_path': '/home/linzhihui/datasets/HumanFace/video/VOS/STCN_PreTrain',
    }
    DATASET.INFO.DAVIS16 = {
        'root_path': '/home/linzhihui/datasets/HumanFace/video/VOS/DAVIS',
        'max_skip': 25,
        'samples_per_video': 5
    }
    DATASET.INFO.DAVIS17 = {
        'root_path': '/home/linzhihui/datasets/HumanFace/video/VOS/DAVIS',
        'max_skip': 25,
        'samples_per_video': 5
    }
    DATASET.INFO.YTVOS18 = {
        'root_path': "/home/linzhihui/datasets/HumanFace/video/VOS/YTVOS18",
        'max_skip': 5,
        'samples_per_video': 1
    }
    DATASET.INFO.YTVOS19 = {
        'root_path': "/home/linzhihui/datasets/HumanFace/video/VOS/YTVOS19",
        'max_skip': 5,
        'samples_per_video': 1
    }
    # ------------------------------------------ dataloader ---------------------------------------------
    DATALOADER = edict()
    DATALOADER.IMG_PER_GPU = 8
    DATALOADER.NUM_WORKERS = 8

    # ----------------------------------------- model configuration ---------------------------------------------
    MODEL = edict()
    MODEL.MODEL_NAME = 'SWEM'
    MODEL.BACKBONE = 'resnet50'            # 'resnet50', 'resnet18'
    MODEL.KEYDIM = 128
    MODEL.VALDIM = 512
    MODEL.NUM_BASES = 256
    MODEL.NUM_EM_ITERS = 4
    MODEL.EM_TAU = 0.05
    MODEL.TOPL = 64
    MODEL.DEVICE = 'cuda'
    MODEL.SINGLE_OBJ = False              # single-object segmentation or not

    # ---------------------------------------- training configuration -------------------------------------------
    SOLVER = edict()
    SOLVER.STAGE = 0           # 0, PRETRAIN; 1, DAVIS17; 2, YTVOS19; 3, DAVIS+YTVOS19
    SOLVER.STAGE_NAME = 'S0'
    SOLVER.BASE_LR = 2e-5
    SOLVER.PRETRAIN_ITERS = [150000, 300000]    # [75000, 150000] for {bs: 32, nobj: 1, lr: 2e-5}, or {bs: 16, nobj: 2, lr: 2e-5}; [150000, 300000] for {bs: 16, nobj: 1, lr: 1e-5} or {bs: 8, nobj: 2, lr: 1e-5}
    SOLVER.MAINTRAIN_ITERS = [125000, 150000]   # [62500, 75000] for {bs: 16, nobj: 2, lr: 2e-5}; [125000, 150000] for {bs: 8, nobj: 2, lr: 1e-5}
    SOLVER.DAVIS_ITERS = [50000, 60000]
    SOLVER.GAMMA = 0.1
    SOLVER.OPTIMIZER = 'AdamW'             # 'SGD' or 'Adam' or 'AdamW'
    SOLVER.MOMENTUM = (0.9, 0.999)
    SOLVER.WEIGHT_DECAY = 5e-4

    # ---------------------------------------- training configuration -------------------------------------------
    LOSS = edict()
    LOSS.NAME = 'boots_ce'            # 'ce' or 'boots_ce'
    LOSS.BS_RATIO = 0.30              # top p%
    LOSS.BS_PERIOD = [20000, 70000]   # start_warm, end_warm
    LOSS.AUX = 'iou'                  # None or 'iou' or 'lovasz'
    LOSS.AUX_RATIO = 1.0
    LOSS.ONLY_VALID_OBJ = DATASET.ONLY_VALID

    # ---------------------------------------- training configuration --------------------------------------------
    RESUME = None  # initialization
    FROM_SCRATCH = True
    AMP = False  # mixture precision
    # ---------------------------------------- testing configuration --------------------------------------------
    VAL = edict()
    VAL.VISUALIZE = False
    # 'DAVIS16', 'DAVIS17', 'YTVOS18'， 'YTVOS19'
    VAL.DAVIS_PALETTE_DIR = '/home/linzhihui/labs/HumanFace/Video/VOS/codes/assets/davis_palette.png'
    VAL.YTVOS_PALETTE_DIR = '/home/linzhihui/labs/HumanFace/Video/VOS/codes/assets/ytvos_palette.png'
    # video datasets
    VAL.DATA_ROOT = {
        'DAVIS16': '/home/linzhihui/datasets/HumanFace/video/VOS/DAVIS',
        'DAVIS17': '/home/linzhihui/datasets/HumanFace/video/VOS/DAVIS',
        'YTVOS18': '/home/linzhihui/datasets/HumanFace/video/VOS/YTVOS18/valid_all_frames',
        'YTVOS19': '/home/linzhihui/datasets/HumanFace/video/VOS/YTVOS19/valid_all_frames',
    }
    # ------------------------------------------- other configuration -------------------------------------------
    CODE_ROOT = '/home/linzhihui/labs/HumanFace/Video/VOS'
    LOG_PERIOD = 100
    SAVE_PERIOD = 5000

    def __init__(self, config_args=None):
        if config_args is not None:
            for key, value in config_args.__dict__.items():
                # for level 1 properties
                if key in self.__class__.__dict__.keys():
                    self.__dict__[key] = value
                elif key in self.DATASET.keys():
                    self.DATASET[key] = value
                elif key in self.DATALOADER.keys():
                    self.DATALOADER[key] = value
                elif key in self.MODEL.keys():
                    self.MODEL[key] = value
                elif key in self.SOLVER.keys():
                    self.SOLVER[key] = value

                if key == 'MAX_NUM_OBJS':
                    self.MODEL.SINGLE_OBJ = (value == 1)
