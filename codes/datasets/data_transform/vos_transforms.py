from __future__ import division

import random
random.seed()
import numpy as np

import torch

from .thinplatespline.batch import TPS
from .thinplatespline.utils import (TOTEN, TOPIL, grid_points_2d, noisy_grid, grid_to_img)


class JointRandomTPS_TEN(object):
    def __init__(self, margin_rate=0.25, num_anchor=(4, 4), p=0.5):
        self.margin_rate = margin_rate
        self.num_anchor = num_anchor
        self.p = p

    @staticmethod
    def get_params(img, margin_rate, num_anchor):
        _, h, w = img.shape
        X = grid_points_2d(num_anchor[1], num_anchor[0], img.device)
        Y = noisy_grid(num_anchor[1], num_anchor[0], margin_rate, img.device)
        tpsb = TPS(size=(h, w), device=img.device)
        warped_grid_b = tpsb(X[None, ...], Y[None, ...])

        return warped_grid_b

    def __call__(self, imgs, msks, other_msks=None):
        """
        :param imgs: images, list of tensors, (C, H, W)
        :param msks: masks, list of tensors, (1, H, W)
        :param other_msks: masks, list of tensors, (1, H, W), optional
        :return:
        tps_imgs, tps_msks, tps_other_msks
        """
        tar_imgs = []
        tar_msks = []
        if other_msks is not None:
            tar_other_msks = []
        else:
            tar_other_msks = None

        for i in range(len(imgs)):
            if random.random() < self.p:
                warped_grid_b = self.get_params(imgs[i], self.margin_rate, self.num_anchor)
                ten_wrp_img = torch.nn.functional.grid_sample(imgs[i][None, ...], warped_grid_b, mode='bilinear')
                tar_imgs.append(ten_wrp_img[0])

                ten_wrp_msk = torch.nn.functional.grid_sample(msks[i][None, ...], warped_grid_b, mode='nearest')
                #print(f'In TPS, Max value of mask is {ten_wrp_msk.numpy().max()}. Min value of mask is {ten_wrp_msk.numpy().min()}')
                tar_msks.append(ten_wrp_msk[0])

                if other_msks is not None:
                    ten_other_msk = other_msks[i]
                    ten_wrp_other_msk = torch.nn.functional.grid_sample(ten_other_msk[None, ...], warped_grid_b,
                                                                        mode='nearest')
                    tar_other_msks.append(ten_wrp_other_msk[0])
            else:
                tar_imgs.append(imgs[i])
                tar_msks.append(msks[i])
                if other_msks is not None:
                    tar_other_msks.append(other_msks[i])
        if tar_other_msks is None:
            return tar_imgs, tar_msks
        else:
            return tar_imgs, tar_msks, tar_other_msks

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'margin_rate={}'.format(self.margin_rate)
        format_string += ', num_anchor={}'.format(self.num_anchor)
        format_string += ', p={})'.format(self.p)
        return format_string


class ToTensor(object):
    def __call__(self, masks):
        masks = [mask[:, :, np.newaxis] if mask.ndim == 2 else mask for mask in masks]
        tar_masks = [torch.from_numpy(mask.transpose([2, 0, 1])) for mask in masks]
        return tar_masks

    def __repr__(self):
        return self.__class__.__name__ + '()'