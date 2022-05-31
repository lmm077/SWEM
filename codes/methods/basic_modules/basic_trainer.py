"""
Author: Zhihui Lin,
Email: linchrist@163.com,
DATE: Y/M/D-2021/08/04
"""

import os
import logging
from PIL import Image
import cv2
import numpy as np
import time
import torch
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import torchvision
from tensorboardX import SummaryWriter
from torch.utils.data import ConcatDataset
from datasets import renew_vos_dataset
from utils import mkdir, setup_logger, AvgMeter, save_overlay, add_overlay


class BasicTrainer:
    def __init__(self, config, name='baseline', is_dist=False, rank=0):
        self.config = config
        # set up dirs and log
        root_dir = config.CODE_ROOT
        log_dir = os.path.join(root_dir, 'logs', config.MODEL.MODEL_NAME, config.SOLVER.STAGE_NAME, name)
        model_dir = os.path.join(log_dir, 'models')
        solver_dir = os.path.join(log_dir, 'solvers')
        self.model_path = os.path.join(model_dir, f'{config.MODEL.MODEL_NAME}.pth')
        self.solver_path = os.path.join(solver_dir, f'{config.MODEL.MODEL_NAME}.solver')

        self.logger = None
        self.tb_writer = None

        self.palette = Image.open(config.VAL.YTVOS_PALETTE_DIR).getpalette()

        if rank <= 0:
            mkdir(log_dir)
            mkdir(model_dir)
            mkdir(solver_dir)

            tb_dir = os.path.join(log_dir, 'tb_log')
            self.tb_writer = SummaryWriter(tb_dir)
            setup_logger('base', log_dir, 'train_stage', level=logging.INFO, screen=False, to_file=True)
            self.logger = logging.getLogger('base')

        # settings
        self.is_dist = is_dist
        self.rank = rank
        self.device = torch.device(config.MODEL.DEVICE)

        self.model = None
        self.optimizer = None
        self.lr_scheduler = None
        self.criterion = None
        self.dataloader = None
        self.skip_iters = []
        self.cur_iter, self.max_iter = 0, 0

        self.single_object = config.MODEL.SINGLE_OBJ

        if rank <= 0:
            self.logger.info(f'+++++++++++++++++++++++++++++++++++++ configs +++++++++++++++++++++++++++++++++++++')
            for key, value in config.__dict__.items():
                self.logger.info(f'{key}: {value}')
            self.logger.info(f'+++++++++++++++++++++++++++++++++++++ dataset +++++++++++++++++++++++++++++++++++++')
            for key, value in config.DATASET.items():
                if not isinstance(value, dict):
                    self.logger.info(f'{key}: {value}')
            self.logger.info(f'++++++++++++++++++++++++++++++++++++ dataloader ++++++++++++++++++++++++++++++++++++')
            for key, value in config.DATALOADER.items():
                self.logger.info(f'{key}: {value}')
            self.logger.info(f'++++++++++++++++++++++++++++++++++++++ model ++++++++++++++++++++++++++++++++++++++')
            for key, value in config.MODEL.items():
                self.logger.info(f'{key}: {value}')
            self.logger.info(f'++++++++++++++++++++++++++++++++++++++ solver ++++++++++++++++++++++++++++++++++++++')
            for key, value in config.SOLVER.items():
                self.logger.info(f'{key}: {value}')

        if self.config.AMP:
            if rank <= 0:
                self.logger.info(f'Training with AMP')
            self.scaler = torch.cuda.amp.GradScaler()

    @staticmethod
    def set_bn_eval(m):
        # only freeze running mean, running val
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.eval()

    @staticmethod
    def set_bn_freeze(m):
        # freeze both running mean, running val and affine parameters
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            for p in m.parameters():
                p.requires_grad = False

            m.eval()

    @staticmethod
    def reduce_tensor(tensor):
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= int(os.environ['WORLD_SIZE'])
        return rt

    def save_model(self):
        if isinstance(self.model, DataParallel) or isinstance(self.model, DistributedDataParallel):
            torch.save(self.model.module.state_dict(), self.model_path)
        else:
            torch.save(self.model.state_dict(), self.model_path)

    def load_model(self, model_path, strict=True, cpu=False):
        if cpu:
            loaded_model = torch.load(model_path, map_location='cpu')
        else:
            map_location = 'cuda:%d' % self.rank
            loaded_model = torch.load(model_path, map_location={'cuda:0': map_location})

        # Maps SO weight (without other_mask) to MO weight (with other_mask)
        for k in list(loaded_model.keys()):
            if k == 'value_encoder.conv1.weight':
                if loaded_model[k].shape[1] == 4 and not self.single_object:
                    pads = torch.zeros((64, 1, 7, 7), device=loaded_model[k].device)
                    torch.nn.init.orthogonal_(pads)
                    loaded_model[k] = torch.cat([loaded_model[k], pads], 1)

        if isinstance(self.model, DataParallel) or isinstance(self.model, DistributedDataParallel):
            self.model.module.load_state_dict(loaded_model, strict=strict)
        else:
            self.model.load_state_dict(loaded_model, strict=strict)

        del loaded_model
        torch.cuda.empty_cache()

    def save_solver(self, iteration, loss):
        solver = dict()
        solver['optimizer'] = self.optimizer.state_dict()
        solver['lr_scheduler'] = self.lr_scheduler.state_dict()
        solver['iteration'] = iteration
        solver['loss'] = loss
        torch.save(solver, self.solver_path)

    def load_solver(self, solver_path):
        loaded_solver = torch.load(solver_path)
        loaded_optimizer = loaded_solver['optimizer']
        loaded_lr_scheduler = loaded_solver['lr_scheduler']
        iteration = loaded_solver['iteration']
        loss = loaded_solver['loss']
        self.optimizer.load_state_dict(loaded_optimizer)
        self.lr_scheduler.load_state_dict(loaded_lr_scheduler)

        del loaded_solver
        torch.cuda.empty_cache()

        return iteration, loss

    def one_step(self, frames, masks, valid_obj, label, cur_iter):
        losses = {}
        results = []
        return losses, results

    def vis_results(self, batch, results):
        images = batch['images']  # B, T, C, H, W
        masks = batch['masks']  # B, T, C, H, W
        label = torch.argmax(masks, dim=2).long()  # B, T, H, W
        results = results.cpu()
        masks = torch.cat([label[:, 0:1], results], dim=1)  # B, T, H, W

        b, t = images.shape[:2]
        overlays = []
        for i in range(b):
            for j in range(t):
                img = (images[i][j].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                msk = (masks[i][j].cpu().numpy().astype(np.uint8))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                overlay_img = add_overlay(img, msk, self.palette, alpha=0.4)
                file_name = batch['info']['frames'][j][i]
                cv2.putText(overlay_img, file_name, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                            cv2.LINE_AA)
                overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)
                overlay_img = torch.from_numpy(overlay_img.astype(np.float32) / 255.0)  # HWC
                overlays.append(overlay_img)

        return torch.stack(overlays, dim=0), t

    def train(self):
        stats = AvgMeter()
        self.logger.info(f'skip iters: {self.skip_iters}')
        stats_skips = {}
        if self.config.SOLVER.STAGE != 0:
            # set skip step for video dataset
            if isinstance(self.dataloader.dataset, ConcatDataset):
                for data_idx, dataset in enumerate(self.dataloader.dataset.datasets):
                    if dataset.data_name not in stats_skips.keys():
                        stats_skips[dataset.data_name] = AvgMeter(50)
            else:
                if self.dataloader.dataset.data_name not in stats_skips.keys():
                    stats_skips[self.dataloader.dataset.data_name] = AvgMeter(50)

        tic = time.time()
        cur_iter = self.cur_iter
        dataloader = iter(self.dataloader)
        while cur_iter < self.max_iter:
            # prepare a batch of data
            batch = next(dataloader)
            frames = batch['images'].cuda(non_blocking=True)            # B, T, C, H, W
            masks = batch['masks'].cuda(non_blocking=True)              # B, T, (N+1), H, W
            if self.config.DATASET.ONLY_VALID:
                valid_obj = batch['valid_obj'].cuda(non_blocking=True)  # B, (N+1) or None
            else:
                valid_obj = None

            label = torch.argmax(masks, dim=2).long()           # B, T, H, W

            # one training step
            with torch.cuda.amp.autocast(enabled=self.config.AMP):
                losses, results = self.one_step(frames, masks[:, 0], valid_obj, label, cur_iter)

            cur_iter += 1
            # renew dataloader for changing the maximum skip interval of sampled frames
            if cur_iter in self.skip_iters:
                self.dataloader = renew_vos_dataset(self.dataloader, self.config, self.logger,
                                                    self.rank, self.is_dist, cur_iter)
                dataloader = iter(self.dataloader)
                self.skip_iters.remove(cur_iter)

            if self.config.SOLVER.STAGE != 0:
                # statistic mean skip interval of sampled frames
                for idx, data_name in enumerate(batch['info']['dataset']):
                    mean_skip = batch['skips'][idx].item()
                    stats_skips[data_name].update(mean_skip)

            # reduce losses if dist
            if self.is_dist:
                for loss_name, loss in losses.items():
                    if loss_name != 'p':
                        losses[loss_name] = self.reduce_tensor(loss)
            stats.update(losses['total_loss'].item())

            if self.rank <= 0:
                # log results
                if (cur_iter - 1) % self.config.LOG_PERIOD == 0 or cur_iter == self.max_iter:

                    cur_skips = {}
                    if self.config.SOLVER.STAGE != 0:
                        # get maximum skip interval of sampled frames
                        if isinstance(self.dataloader.dataset, ConcatDataset):
                            for data_idx, dataset in enumerate(self.dataloader.dataset.datasets):
                                cur_skips[dataset.data_name] = self.dataloader.dataset.datasets[data_idx].cur_skip
                        else:
                            cur_skips[self.dataloader.dataset.data_name] = self.dataloader.dataset.cur_skip

                    loss_str = ''
                    for loss_name, loss in losses.items():
                        if loss_name != 'p':
                            loss = loss.item()
                        self.tb_writer.add_scalar(f'Train/{loss_name}', loss, cur_iter)
                        loss_str += f'{loss_name}: {loss:.5f}. '
                    current_lr = self.optimizer.param_groups[0]['lr']
                    self.tb_writer.add_scalar('Train/learning_rate', current_lr, cur_iter)

                    intervals = ''
                    for key, value in cur_skips.items():
                        self.tb_writer.add_scalar(f'Data/max_interval_{key}', value, cur_iter)
                        self.tb_writer.add_scalar(f'Data/mean_interval_{key}', stats_skips[key].avg, cur_iter)
                        intervals += f'({key}:{value:02d}|{stats_skips[key].avg:2.2f})'

                    # vis results
                    if (cur_iter - 1) % (self.config.LOG_PERIOD * 10) == 0 or cur_iter == self.max_iter:
                        vis, n_row = self.vis_results(batch, results)
                        # print(f'visualized data szie {vis.size()}, time steps {n_row}')
                        vis = vis.permute(0, 3, 1, 2)  # B, H, W, C -> B, C, H, W
                        vis = torch.nn.functional.interpolate(vis, size=(192, 192), mode='bilinear',
                                                              align_corners=False)
                        vis = torchvision.utils.make_grid(vis, nrow=n_row)
                        self.tb_writer.add_image('Train/seg', vis, cur_iter, dataformats='CHW')

                    # print information
                    total_time = time.time() - tic
                    tic = time.time()
                    iter_time = total_time / self.config.LOG_PERIOD
                    remain_time = round((self.max_iter - cur_iter) * iter_time)
                    remain_time_sec = remain_time % 60
                    remain_time_min = remain_time // 60
                    remain_time_hour = remain_time_min // 60
                    remain_time_min = remain_time_min % 60

                    log_str = f'[Iter: {cur_iter:06d}/{self.max_iter:06d}.' \
                              f' ETA: {remain_time_hour:02d}:{remain_time_min:02d}:{remain_time_sec:02d}.] '
                    log_str += f'Interval: {intervals}. ' + f'LR: {current_lr}. ' + loss_str + f'(AVG: {stats.avg:.5f})'
                    print(log_str)
                    self.logger.info(log_str)

                # save model and solver
                if cur_iter % self.config.SAVE_PERIOD == 0 or cur_iter == self.max_iter:
                    self.save_model()
                    self.save_solver(cur_iter, losses['total_loss'].item())

        if self.rank <= 0:
            self.tb_writer.close()
            self.logger.info('Training done.')

    def val(self):
        pass
