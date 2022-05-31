import os
import numpy as np
import time
from datetime import datetime
import logging

import torch


class Logger(object):

    DefaultItemCount = 1

    def __init__(self, fpath, resume=False):

        mode = 'a' if resume else 'w'
        self.file = open(fpath, mode)
        self.items = []
        self.vals = []

    def close(self):

        self.file.close()
        self.items = []
        self.vals = []

    def set_items(self, item_names=None):

        if item_names is None:
            self.items.append('term %d' % self.DefaultItemCount)
            self.DefaultItemCount += 1
        elif isinstance(item_names, list):
            for item_name in item_names:
                self.items.append(item_name)

    def log(self, *terms):

        assert len(terms) == len(self.items), 'mismatch logger information'

        self.file.write('==> log info time: %s' % time.ctime())
        self.file.write('\n')

        log = ''
        for item, val in zip(self.items, terms):
            if isinstance(val, float):
                formats = '%s %.5f '
            else:
                formats = '%s %d '

            log += formats % (item, val)

        self.file.write(log)

        self.file.write('\n')


class AvgMeter(object):

    def __init__(self, window=-1):
        self.window = window
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0
        self.max = -np.Inf

        if self.window > 0:
            self.val_arr = np.zeros(self.window)
            self.arr_idx = 0

    def update(self, val, n=1):

        self.cnt += n
        self.max = max(self.max, val)

        if self.window > 0:
            self.val_arr[self.arr_idx] = val
            self.arr_idx = (self.arr_idx + 1) % self.window
            self.avg = self.val_arr.mean()
        else:
            self.sum += val * n
            self.avg = self.sum / self.cnt


class FrameSecondMeter(object):

    def __init__(self):
        self.st = time.time()
        self.fps = None
        self.fps_ = None
        self.ti = time.time()
        self.ed = None
        self.frame_n = 0
        self.total_time = 1e-12

    def toc(self, frame_n):
        self.frame_n += frame_n
        self.total_time += (time.time() - self.ti)

    def tic(self):
        self.ti = time.time()

    def end(self):
        self.ed = time.time()
        self.fps_ = self.frame_n / (self.ed - self.st)
        self.fps = self.frame_n / self.total_time


def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')


def setup_logger(logger_name, save_dir, phase, level=logging.INFO, screen=False, to_file=False):
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    lg.setLevel(level)
    if to_file:
        log_file = os.path.join(save_dir, phase + '_{}.log'.format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)


def print_mem(info=None):
    if info:
        print(info, end=' ')
    mem_allocated = round(torch.cuda.memory_allocated() / 1048576)
    mem_cached = round(torch.cuda.memory_cached() / 1048576)
    print(f'Mem allocated: {mem_allocated}MB, Mem cached: {mem_cached}MB')