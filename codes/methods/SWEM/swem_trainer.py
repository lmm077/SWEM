"""
Author: Zhihui Lin,
Email: linchrist@163.com,
DATE: Y/M/D-2021/08/04
"""

import os

import torch
from torch.nn.parallel import DistributedDataParallel

from ..basic_modules.basic_trainer import BasicTrainer
from .swem import SWEM
from solver import get_solver
from datasets import get_vos_dataset
from losses import get_criterion


class SWEMTrainer(BasicTrainer):
    def __init__(self, config, *args, **kwargs):
        super(SWEMTrainer, self).__init__(config, *args, **kwargs)
        # model
        self.model = SWEM(self.config.MODEL)
        if self.rank <= 0:
            self.logger.info(f'MODEL_PARAM: {self.config.MODEL}')
            self.logger.info(f'############################# MODEL STRUCTURE ############################# ')
            self.logger.info(self.model)
            self.logger.info(f'############################# MODEL STRUCTURE ############################# ')
        # load pretrained weights
        if self.config.RESUME is not None:
            resume_path = os.path.join(self.config.RESUME, f'models/{self.config.MODEL.MODEL_NAME}.pth')
            if self.rank <= 0:
                self.logger.info(f'Loading model from {self.config.RESUME}...')
            self.load_model(resume_path, strict=True, cpu=True)
        # model setting
        self.model.to(self.device)
        self.model.train()
        # freeze bn
        self.model.apply(self.set_bn_eval)

        if self.is_dist:
            self.model = DistributedDataParallel(self.model, device_ids=[self.rank], output_device=self.rank,
                                                 broadcast_buffers=False)
        # solver
        self.optimizer, self.lr_scheduler, self.cur_iter, best_loss = get_solver(self.config, self.model,
                                                                                 self.logger, self.rank)
        # dataloader
        self.dataloader, self.max_iter, self.skip_iters = get_vos_dataset(self.config, self.logger,
                                                                          self.rank, self.is_dist,
                                                                          phase='train', cur_iter=self.cur_iter)
        # criterion
        self.criterion = get_criterion(self.config.LOSS, self.logger, self.rank, self.max_iter, self.device)

        if self.rank <= 0:
            print('Construction of SWEM trainer is finished!')
            if self.logger is not None:
                self.logger.info('Construction of SWEM trainer is finished!')

    def one_step(self, frames, init_mask, valid_obj, label, cur_iter):
        b, t, c, h, w = frames.shape
        out_size = init_mask.shape[-2:]

        # initialization memory
        mk16, _, s16, _, _ = self.model('encode_key', frames[:, 0])
        mv16 = self.model('encode_value', frames[:, 0], init_mask.float(), s16)
        _ = self.model('init', mk16, mv16, init_mask)

        logits_list = []
        results = []
        for i in range(1, t):
            # encode
            qk16, qv16, s16, s8, s4 = self.model('encode_key', frames[:, i])
            # matching
            context, n = self.model('match', qk16, qv16)
            # decoding
            logits, pred_mask = self.model('segment', n, context, s8, s4, valid_obj, out_size)
            logits_list.append(logits)

            # get hard mask [0, 1] for memorizing
            pred = torch.argmax(pred_mask, dim=1, keepdim=True)                          # B, 1, H_o, W_o
            results.append(pred)
            pred_expand = pred.expand(-1, n + 1, -1, -1)                                 # B, (n+1), H_o, W_o
            obj_idx = torch.arange(n + 1).type(pred.dtype).to(pred.device)               # (n+1)
            obj_idx = obj_idx.view(1, -1, 1, 1).expand(b, -1, out_size[0], out_size[1])  # B, (n+1), H_o, W_o
            hard_pred_mask = (pred_expand == obj_idx).type_as(pred)                      # B, (n+1), H_o, W_o
            # memorizing if need
            if i < t - 1:
                # encode memory
                mv16 = self.model('encode_value', frames[:, i], pred_mask, s16)  # BN, C, H, W
                _ = self.model('memorize', qk16, mv16, hard_pred_mask, pred_mask)

        logits = torch.stack(logits_list, dim=2)                                         # B, N+1, T, H, W
        results = torch.cat(results, dim=1)
        # get losses and backward
        losses = self.criterion(logits, label[:, 1:], cur_iter, valid_obj=valid_obj)
        total_loss = losses['total_loss']

        self.optimizer.zero_grad(set_to_none=True)
        if self.config.AMP:
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            self.optimizer.step()
        self.lr_scheduler.step()

        return losses, results

    def val(self):
        pass
