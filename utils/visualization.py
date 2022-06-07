import numpy as np
from PIL import Image
from scipy.ndimage.morphology import binary_dilation
import cv2

import torch


def calc_uncertainty(score):
    # seg shape: bs, obj_n, h, w
    score_top, _ = score.topk(k=2, dim=1)
    uncertainty = score_top[:, 0] / (score_top[:, 1] + 1e-8)  # bs, h, w
    uncertainty = torch.exp(1 - uncertainty).unsqueeze(1)  # bs, 1, h, w
    return uncertainty


def save_seg(seg, obj_labels, save_path):
    img = np.zeros((*seg.shape, 3), dtype=np.uint8)
    for idx, label_id in enumerate(obj_labels):
        img[(seg == idx)] = label_id[::-1]  # RGB -> BGR
    cv2.imwrite(save_path, img)


def save_heatmap(overlay_path, img, heat):
    # heat, H x W, [0, 1]
    # img, H x W x 3 [0, 255]
    img = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    heat = (1 - heat).cpu().numpy().astype(np.float32)
    heat = (heat - np.min(heat)) / (np.max(heat) - np.min(heat) + 1e-8)  # norm
    heat = (heat * 255).astype(np.uint8)
    heat = cv2.resize(heat, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
    heat_img = cv2.applyColorMap(heat, cv2.COLORMAP_JET)                 # BGR
    img_add = cv2.addWeighted(img, 0.7, heat_img, 0.3, 0)

    cv2.imwrite(overlay_path, img_add)


def save_seg_mask(pred, seg_path, palette):
    seg_img = Image.fromarray(pred)
    seg_img.putpalette(palette)
    seg_img.save(seg_path)


def add_overlay(img, mask, colors, alpha=0.4, cscale=1):
    ids = np.unique(mask)
    img_overlay = img.copy()
    ones_np = np.ones(img.shape) * (1 - alpha)

    colors = np.reshape(colors, (-1, 3))
    colors = np.atleast_2d(colors) * cscale

    for i in ids[1:]:

        canvas = img * alpha + ones_np * np.array(colors[i])[::-1]

        binary_mask = mask == i
        img_overlay[binary_mask] = canvas[binary_mask]

        contour = binary_dilation(binary_mask) ^ binary_mask
        img_overlay[contour, :] = 0

    return img_overlay


def save_overlay(img, mask, overlay_path, colors=[255, 0, 0], alpha=0.4, cscale=1):
    img = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    img_overlay = add_overlay(img, mask, colors, alpha, cscale)
    cv2.imwrite(overlay_path, img_overlay)


# visualization for hard sample mining
def save_hard_sample_masks(img, mask, save_path, alpha=0.3):
    # hard_sample_mask， h x w
    img = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]

    mask = cv2.resize(mask, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
    mask = (mask * 255).astype(np.uint8)
    heat_map = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    heat_map = cv2.addWeighted(heat_map, alpha, img, 1-alpha, 0)

    cv2.imwrite(save_path, heat_map)



