"""
Author: Zhihui Lin,
Email: linchrist@163.com,
DATE: Y/M/D-2021/08/04
"""

import torch

import torch.nn.functional as F
import time

from ..basic_modules.basic_evaluator import BasicEvaluator
from .swem import SWEM


class SWEMEvaluator(BasicEvaluator):
    def __init__(self, config, *args, **kwargs):
        super(SWEMEvaluator, self).__init__(config, *args, **kwargs)
        self.config.MODEL.IMAGE_PRETRAIN = False
        self.model = SWEM(self.config.MODEL)
        self.logger.info(f'MODEL_PARAM: {self.config.MODEL}')
        self.logger.info(f'############################# MODEL STRUCTURE ############################# ')
        self.logger.info(self.model)
        self.logger.info(f'############################# MODEL STRUCTURE ############################# ')

        if self.config.RESUME is not None:
            self.logger.info(f'Loading model from {self.config.RESUME}...')
            self.load_model(self.config.RESUME, strict=True, cpu=True)

        self.model.to(self.device)
        self.model.eval()
        self.logger.info('Total params: %.2fM' % (sum(p.numel() for p in self.model.parameters()) / 1000000.0))

    def evaluate_davis_seq_ms(self, frames, init_masks, out_size, scales=[480], is_flip=False):
        # frames, tensor, (1, T, C, H, W)
        # init_masks, list of (1, N+1, H, W)
        # prev_mask, tensor (1, N+1, H, W)
        final_pred_scores = [0 for _ in range(frames.shape[1]-1)]
        assert (len(scales) > 0)
        for scale in scales:
            h = scale
            w = int((scale / 480) * 864)
            in_frames = F.interpolate(frames[0], size=(h, w), mode='bicubic', align_corners=False).unsqueeze(0)
            preds, pred_scores = self.evaluate_davis_seq(in_frames, init_masks, out_size)
            if is_flip:
                in_frames_flip = torch.flip(in_frames, dims=[-1])
                init_masks_flip = [torch.flip(init_mask, dims=[-1]) for init_mask in init_masks if init_mask is not None]
                pred_scores_flip = self.evaluate_davis_seq(in_frames_flip, init_masks_flip, out_size)[-1]
                pred_scores = [(pred_score + torch.flip(pred_score_flip, dims=[-1]))/2 for pred_score, pred_score_flip
                               in zip(pred_scores, pred_scores_flip)]
                
            final_pred_scores = [final_pred_score + pred_score / len(scales) for
                                 final_pred_score, pred_score in zip(final_pred_scores, pred_scores)]

        final_preds = [torch.argmax(final_pred_score, dim=1, keepdim=False) for final_pred_score in final_pred_scores]  # list of B, H_o, W_o

        return final_preds

    def evaluate_davis_seq(self, frames, init_masks, out_size):
        preds = []
        pred_scores = []
        sec_per_frame = '[ '
        b, t, c, h, w = frames.shape
        # initialization memory
        tic = time.time()
        mk16, _, s16, _, _ = self.model('encode_key', frames[:, 0])
        init_mask = F.interpolate(init_masks[0], size=(h, w), mode='nearest')
        mv16 = self.model('encode_value', frames[:, 0], init_mask.float(), s16)
        self.model('init', mk16, mv16, init_masks[0])
        sec_per_frame += f'{time.time() - tic} '

        for i in range(1, t):
            tic = time.time()
            # encode
            qk16, qv16, s16, s8, s4 = self.model('encode_key', frames[:, i])
            # matching
            context, n = self.model('match', qk16, qv16)
            # decoding
            logits, pred_mask = self.model('segment', n, context, s8, s4, None, out_size)
            pred_scores.append(pred_mask.clone())

            # get hard mask [0, 1] for memorizing
            pred = torch.argmax(pred_mask, dim=1, keepdim=True)  # B, 1, H_o, W_o
            pred_expand = pred.expand(-1, n + 1, -1, -1)  # B, (n+1), H_o, W_o
            obj_idx = torch.arange(n + 1).type(pred.dtype).to(pred.device)  # (n+1)
            obj_idx = obj_idx.view(1, -1, 1, 1).expand(b, -1, out_size[0], out_size[1])  # B, (n+1), H_o, W_o
            hard_pred_mask = (pred_expand == obj_idx).type_as(pred)                      # B, (n+1), H_o, W_o

            # memorizing if need
            if i < t - 1:
                pred_mask = F.interpolate(pred_mask, size=(h, w), mode='bilinear', align_corners=False)
                mv16 = self.model('encode_value', frames[:, i], pred_mask, s16)  # BN, C, H, W
                self.model('memorize', qk16, mv16, hard_pred_mask, pred_mask)

            sec_per_frame += f'{time.time() - tic} '

            preds.append(pred[:, 0])

        sec_per_frame += ']'
        self.logger.info(sec_per_frame)

        return preds, pred_scores

    def evaluate_ytvos_seq(self, frames, init_masks, out_size):
        preds = []
        # need to be overritten
        b, t, c, h, w = frames.shape
        # initialization memory
        mk16, _, s16, _, _ = self.model('encode_key', frames[:, 0])
        init_mask = F.interpolate(init_masks[0], size=(h, w), mode='nearest')
        mv16 = self.model('encode_value', frames[:, 0], init_mask.float(), s16)
        self.model('init', mk16, mv16, init_masks[0])

        # for i in range(1, t)
        for i in range(1, t):
            # encode
            qk16, qv16, s16, s8, s4 = self.model('encode_key', frames[:, i])
            # matching
            context, n = self.model('match', qk16, qv16)
            # decoding
            logits, pred_mask = self.model('segment', n, context, s8, s4, None, out_size)

            # add new masks if new objects appear, for YT-VOS data
            if init_masks[i] is not None:
                # new_masks, B, N'+1, H_o, W_o
                new_objects = torch.sum(init_masks[i][:, 1:], dim=1, keepdim=True)  # B, 1, H_o, W_o
                new_objects = new_objects.expand_as(pred_mask)                      # B, N+1, H_o, W_o
                pred_mask[new_objects > 0] = 0
                pred_mask = torch.cat([pred_mask, init_masks[i][:, 1:]], dim=1)     # B, N+N'+1, H_o, W_o
                n = pred_mask.shape[1] - 1

            # get hard mask [0, 1] for memorizing
            pred = torch.argmax(pred_mask, dim=1, keepdim=True)  # B, 1, H_o, W_o
            pred_expand = pred.expand(-1, n + 1, -1, -1)  # B, (n+1), H_o, W_o
            obj_idx = torch.arange(n + 1).type(pred.dtype).to(pred.device)  # (n+1)
            obj_idx = obj_idx.view(1, -1, 1, 1).expand(b, -1, out_size[0], out_size[1])  # B, (n+1), H_o, W_o
            hard_pred_mask = (pred_expand == obj_idx).type_as(pred)                      # B, (n+1), H_o, W_o
            # memorizing if need
            if i < t - 1:
                pred_mask = F.interpolate(pred_mask, size=(h, w), mode='bilinear', align_corners=False)
                mv16 = self.model('encode_value', frames[:, i], pred_mask, s16)  # BN, C, H, W
                self.model('memorize', qk16, mv16, hard_pred_mask, pred_mask)

            preds.append(pred[:, 0])

            #prev_mask = hard_pred_mask

        return preds

    def val(self):
        if 'DAVIS' in self.eval_set:
            self.evaluate_davis()
            if self.eval_set not in ['DAVIS17Test']:
                self.get_metrics()
        elif self.eval_set in ['YTVOS18', 'YTVOS19']:
            self.evaluate_ytvos()
        else:
            raise ValueError(f'dataset {self.eval_set} is not supported yet.')

        self.logger.info('Evaluation done.')
