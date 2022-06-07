"""
Author: Zhihui Lin,
Email: linchrist@163.com,
DATE: Y/M/D-2021/08/04
"""

import os
import sys
import time
import pandas as pd
from PIL import Image
from tqdm import tqdm as tq
import logging
import numpy as np
import torch
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import DAVIS_Test, YTVOS_Test
from utils import mkdir, init_random_seed, setup_logger, FrameSecondMeter, save_seg_mask, save_overlay


class BasicEvaluator:
    def __init__(self, config, name='baseline', eval_set='DAVIS16', rsize=480, clip_len=32):
        self.config = config
        self.device = torch.device(config.MODEL.DEVICE)
        # set up dirs and log
        root_dir = config.CODE_ROOT
        log_dir = os.path.join(root_dir, 'logs', config.MODEL.MODEL_NAME, config.SOLVER.STAGE_NAME, name)
        self.save_dir = os.path.join(log_dir, 'results', eval_set)
        mkdir(self.save_dir)
        self.eval_set = eval_set

        setup_logger('base', self.save_dir, 'test_stage', level=logging.INFO, screen=True, to_file=True)
        self.logger = logging.getLogger('base')
        self.logger.info(f'Save results in {self.save_dir}')
        # set up random seed
        init_random_seed(config.DATASET.SEED)
        self.single_object = config.MODEL.SINGLE_OBJ

        # model path
        if self.config.RESUME is None:
            self.config.RESUME = os.path.join(log_dir, 'models', f'{config.MODEL.MODEL_NAME}.pth')
        self.model = None

        # dataset
        # define dataset
        data_root = config.VAL.DATA_ROOT[eval_set]
        if eval_set == 'DAVIS16':
            dataset = DAVIS_Test(data_root, img_set='2016/val.txt', single_obj=True)
        elif eval_set == 'DAVIS17':
            dataset = DAVIS_Test(data_root, img_set='2017/val.txt', single_obj=False)
        elif eval_set == 'DAVIS17Test':
            dataset = DAVIS_Test(data_root, img_set='2017/test-dev.txt', single_obj=False)
        elif eval_set in ['YTVOS18', 'YTVOS19']:
            dataset = YTVOS_Test(data_root, dataset_file='meta.json', short_size=rsize)
        else:
            raise ValueError(f'{eval_set} is unsupported yet.')

        self.dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

        # palette
        self.ytvos_palette = Image.open(config.VAL.YTVOS_PALETTE_DIR).getpalette()
        self.davis_palette = Image.open(config.VAL.DAVIS_PALETTE_DIR).getpalette()

        self.rsize = rsize
        self.clip_len = clip_len  # for ytvos data
        self.vis = self.config.VAL.VISUALIZE

        self.dataset_eval = None
        if eval_set == 'DAVIS16':
            from evaluation.davis2017.evaluation import DAVISEvaluation
            # Create dataset and evaluate
            self.dataset_eval = DAVISEvaluation(
                davis_root=self.config.DATASET.INFO.DAVIS16['root_path'],
                year='2016',
                task='semi-supervised',
                gt_set='val',
                store_results=False,
                res_root=self.save_dir+'/output')
        elif eval_set == 'DAVIS17':
            from evaluation.davis2017.evaluation import DAVISEvaluation
            self.dataset_eval = DAVISEvaluation(
                davis_root=self.config.DATASET.INFO.DAVIS17['root_path'],
                year='2017',
                task='semi-supervised',
                gt_set='val',
                store_results=False,
                res_root=self.save_dir + '/output')

        self.logger.info(f'+++++++++++++++++++++++++++++++++++++ configs +++++++++++++++++++++++++++++++++++++')
        for key, value in config.__dict__.items():
            self.logger.info(f'{key}: {value}')
        self.logger.info(f'+++++++++++++++++++++++++++++++++++++ dataset +++++++++++++++++++++++++++++++++++++')
        for key, value in config.DATASET.items():
            if not isinstance(value, dict):
                self.logger.info(f'{key}: {value}')
        self.logger.info(f'++++++++++++++++++++++++++++++++++++++ model ++++++++++++++++++++++++++++++++++++++')
        for key, value in config.MODEL.items():
            self.logger.info(f'{key}: {value}')

    def load_model(self, model_path, strict=True, cpu=False):
        if cpu:
            loaded_model = torch.load(model_path, map_location='cpu')
        else:
            loaded_model = torch.load(model_path)

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

    def evaluate_davis_seq_ms(self, frames, init_mask, out_size, scales, is_flip):
        # need to be overritten
        return None

    def evaluate_davis_seq(self, frames, init_mask, out_size):
        # need to be overritten
        return None, None

    def evaluate_ytvos_seq(self, frames, init_mask, out_size):
        # need to be overritten
        return None

    def mkdirs(self, seq_name):
        out_dir = os.path.join(self.save_dir, 'output', seq_name)
        mkdir(out_dir)

        if self.vis:
            overlay_dir = os.path.join(self.save_dir, 'overlay', seq_name)
            mkdir(overlay_dir)
            return out_dir, overlay_dir
        else:
            return out_dir, out_dir

    def evaluate_davis(self):
        fps = FrameSecondMeter()

        for seq_idx, batch in enumerate(self.dataloader):
            seq_name = batch['info']['name'][0]
            obj_n = batch['info']['obj_n'][0].item()
            frame_n = batch['info']['num_frames'][0].item()

            frames = batch['images'].to(self.device)  # 1 x T x C x H x W
            masks = batch['masks'].to(self.device)  # 1 x 1 x (N+1) x H x W
            # resize
            in_frames = F.interpolate(frames[0], size=(480, 864), mode='bicubic', align_corners=False).unsqueeze(0)
            in_masks = [None for _ in range(frames.size(1))]
            in_masks[0] = masks[:, 0].float()
            # make dirs for saving results
            out_dir, overlay_dir = self.mkdirs(seq_name)

            self.logger.info(f'Testing video {seq_idx}: {seq_name}')

            # pred 1~frame_n masks
            tar_size = frames.size()[-2:]

            with torch.no_grad():
                torch.cuda.synchronize()
                fps.tic()
                preds, scores = self.evaluate_davis_seq(in_frames, in_masks, out_size=tar_size)
                torch.cuda.synchronize()
                fps.toc(frame_n)
                preds = torch.cat(preds, dim=0).cpu().numpy().astype(np.uint8)        # (T-1) x H x W
            # empty cache
            torch.cuda.empty_cache()
            # save mask 0
            pred = torch.argmax(masks[0, 0], dim=0).cpu().numpy().astype(np.uint8)
            seg_path = os.path.join(out_dir, '00000.png')
            save_seg_mask(pred, seg_path, self.davis_palette)

            if self.vis:
                overlay_path = os.path.join(overlay_dir, '00000.png')
                save_overlay(frames[0][0], pred, overlay_path, self.davis_palette)

            for t in tq(range(1, frame_n), desc=f'{seq_idx} {seq_name}'):
                pred = preds[t - 1]
                seg_path = os.path.join(out_dir, f'{t:05d}.png')
                save_seg_mask(pred, seg_path, self.davis_palette)
                if self.vis:
                    # vis for results
                    overlay_path = os.path.join(overlay_dir, f'{t:05d}.png')
                    save_overlay(frames[0][t], pred, overlay_path, self.davis_palette)

        fps.end()
        self.logger.info(f'fps: {fps.fps}')

    @staticmethod
    def map_mask(mask, obj_idx_ten):
        pred = torch.zeros_like(mask)    # H x W
        for i in range(obj_idx_ten.shape[0]):
            pred[mask == i] = obj_idx_ten[i]
        return pred.cpu().numpy().astype(np.uint8)

    def evaluate_ytvos(self):
        seq_n = len(self.dataloader)
        fps = FrameSecondMeter()

        for seq_idx, batch in enumerate(self.dataloader):
            obj_n = batch['info']['obj_n'][0]
            info = batch['info']
            frames = batch['images'].to(self.device)  # 1 x T x C x H x W
            init_masks = [None for _ in range(frames.shape[1])]
            for frame_idx, masks in batch['init_masks'].items():
                init_masks[frame_idx] = masks[0].to(self.device).float()  # list of 1 x (N+1) x H x W

            frame_n = info['num_frames'][0].item()
            seq_name = info['name'][0]
            obj_n = obj_n.item()
            obj_st = [info['obj_st'][0, i].item() for i in range(obj_n)]
            obj_idx_ten = info['obj_idx_ten'][0].to(self.device)
            basename_list = [info['basename_list'][i][0] for i in range(frame_n)]
            basename_to_save = [info['basename_to_save'][i][0] for i in range(len(info['basename_to_save']))]
            obj_vis = info['obj_vis'][0]
            original_size = (info['original_size'][0].item(), info['original_size'][1].item())

            self.logger.info(f'Video {seq_name}, the original size is {original_size}, input size is {frames.shape[3:]}.')

            out_dir, overlay_dir = self.mkdirs(seq_name)
            # log
            self.logger.info(f'Testing video {seq_name}')

            # Compose the first mask
            # save mask 0
            first_mask = batch['first_mask'][0].numpy().astype(np.uint8)
            seg_path = os.path.join(out_dir, basename_list[0] + '.png')
            save_seg_mask(first_mask, seg_path, self.ytvos_palette)
            if self.vis:
                frame_out = F.interpolate(frames[0][0].unsqueeze(0), original_size).squeeze(0)
                overlay_path = os.path.join(overlay_dir, basename_list[0] + '.png')
                save_overlay(frame_out, first_mask, overlay_path, self.ytvos_palette)

            # pred 1~frame_n masks
            with torch.no_grad():
                torch.cuda.synchronize()
                fps.tic()
                preds = self.evaluate_ytvos_seq(frames, init_masks, out_size=original_size)
                torch.cuda.synchronize()
                fps.toc(frame_n)

            for t in tq(range(1, frame_n), desc=f'{seq_idx:3d}/{seq_n:3d} {seq_name}'):
                pred = self.map_mask(preds[t - 1][0], obj_idx_ten)

                # save all results or only save appeared in valid set
                if basename_list[t] in basename_to_save:
                    seg_path = os.path.join(out_dir, basename_list[t] + '.png')
                    save_seg_mask(pred, seg_path, self.ytvos_palette)

                if self.vis:
                    frame_out = F.interpolate(frames[0][t].unsqueeze(0), original_size).squeeze(0)
                    overlay_path = os.path.join(overlay_dir, basename_list[t] + '.png')
                    save_overlay(frame_out, pred, overlay_path, self.ytvos_palette)

            # empty cache
            del preds
            torch.cuda.empty_cache()

    def get_metrics(self):
        time_start = time.time()
        # Check if the method has been evaluated before, if so read the results, otherwise compute the results
        csv_name_global = f'global_results-{self.eval_set}.csv'
        csv_name_per_sequence = f'per-sequence_results-{self.eval_set}.csv'
        csv_name_global_path = os.path.join(self.save_dir, csv_name_global)
        csv_name_per_sequence_path = os.path.join(self.save_dir, csv_name_per_sequence)
        if os.path.exists(csv_name_global_path) and os.path.exists(csv_name_per_sequence_path):
            old_csv_name_global_path = os.path.join(self.save_dir, f'old_{csv_name_global}')
            old_csv_name_per_sequence_path = os.path.join(self.save_dir, f'old_{csv_name_per_sequence}')
            print(
                f'Rename precomputed results as {old_csv_name_global_path} and old_{old_csv_name_per_sequence_path}...')
            os.system(f'mv {csv_name_global_path} {old_csv_name_global_path}')
            os.system(f'mv {csv_name_per_sequence_path} {old_csv_name_per_sequence_path}')

        metrics_res = self.dataset_eval.evaluate(res_path=self.save_dir+'/output')
        J, F = metrics_res['J'], metrics_res['F']

        # Generate dataframe for the general results
        g_measures = ['J&F-Mean', 'J-Mean', 'J-Recall', 'J-Decay', 'F-Mean', 'F-Recall', 'F-Decay']
        final_mean = (np.mean(J["M"]) + np.mean(F["M"])) / 2.
        g_res = np.array(
            [final_mean, np.mean(J["M"]), np.mean(J["R"]), np.mean(J["D"]), np.mean(F["M"]), np.mean(F["R"]),
             np.mean(F["D"])])
        g_res = np.reshape(g_res, [1, len(g_res)])
        table_g = pd.DataFrame(data=g_res, columns=g_measures)
        with open(csv_name_global_path, 'w') as f:
            table_g.to_csv(f, index=False, float_format="%.3f")
        print(f'Global results saved in {csv_name_global_path}')

        # Generate a dataframe for the per sequence results
        seq_names = list(J['M_per_object'].keys())
        seq_measures = ['Sequence', 'J-Mean', 'F-Mean']
        J_per_object = [J['M_per_object'][x] for x in seq_names]
        F_per_object = [F['M_per_object'][x] for x in seq_names]
        table_seq = pd.DataFrame(data=list(zip(seq_names, J_per_object, F_per_object)), columns=seq_measures)
        with open(csv_name_per_sequence_path, 'w') as f:
            table_seq.to_csv(f, index=False, float_format="%.3f")
        print(f'Per-sequence results saved in {csv_name_per_sequence_path}')

        # logging per object per frame results
        for seq_name in seq_names:
            J_per_obj_frame = J['per_obj_frame'][seq_name].tolist()
            F_per_obj_frame = F['per_obj_frame'][seq_name].tolist()
            self.logger.info(f'Per obj per frame J, {seq_name}:{J_per_obj_frame}')
            self.logger.info(f'Per obj per frame F, {seq_name}:{F_per_obj_frame}')

        # Print the results
        sys.stdout.write(f"--------------------------- Global results for {self.eval_set} ---------------------------\n")
        print(table_g.to_string(index=False))
        self.logger.info(f"--------------------------- Global results for {self.eval_set} ---------------------------\n")
        self.logger.info(table_g.to_string(index=False))
        sys.stdout.write(f"\n---------- Per sequence results for {self.eval_set} ----------\n")
        print(table_seq.to_string(index=False))
        self.logger.info(f"\n---------- Per sequence results for {self.eval_set} ----------\n")
        self.logger.info(table_seq.to_string(index=False))
        total_time = time.time() - time_start
        sys.stdout.write('\nTotal time:' + str(total_time))

    def val(self):
        pass
