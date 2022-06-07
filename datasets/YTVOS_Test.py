import os
import json
import numpy as np
from glob import glob
from itertools import compress

import torch
from torch.utils import data

from .data_transform import ToTensor
from .data_utils import load_image_cv, load_anno_np


def get_suit_size(size, ratio=16):
    r = size % ratio
    size -= r
    if r > 7:
        size += 16
    return size


class YTVOS_Test(data.Dataset):
    def __init__(self, root, dataset_file='meta.json', short_size=495, max_obj_n=11):
        self.root = root
        self.max_obj_n = max_obj_n

        self.sszie = get_suit_size(short_size)

        dataset_path = os.path.join(root, dataset_file)
        with open(dataset_path, 'r') as json_file:
            self.meta_data = json.load(json_file)

        self.dataset_list = list(self.meta_data['videos'])
        self.dataset_size = len(self.dataset_list)

        self.to_tensor = ToTensor()

    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, idx):

        video_name = self.dataset_list[idx]

        img_dir = os.path.join(self.root, 'JPEGImages', video_name)
        mask_dir = os.path.join(self.root, 'Annotations', video_name)

        img_list = sorted(glob(os.path.join(img_dir, '*.jpg')))
        basename_list = [os.path.basename(x)[:-4] for x in img_list]
        video_len = len(img_list)
        selected_idx = np.ones(video_len, np.bool)

        objs = self.meta_data['videos'][video_name]['objects']
        obj_n = 1
        video_obj_appear_st_idx = video_len

        for obj_idx, obj_gt in objs.items():
            obj_n = max(obj_n, int(obj_idx) + 1)
            video_obj_appear_idx = basename_list.index(obj_gt['frames'][0])
            video_obj_appear_st_idx = min(video_obj_appear_st_idx, video_obj_appear_idx)

        selected_idx[:video_obj_appear_st_idx] = False
        selected_idx = selected_idx.tolist()
        # remove the frames since objects appear
        img_list = list(compress(img_list, selected_idx))
        basename_list = list(compress(basename_list, selected_idx))

        video_len = len(img_list)
        obj_vis = np.zeros((video_len, obj_n), np.uint8)
        obj_vis[:, 0] = 1
        obj_st = np.zeros(obj_n, np.uint8)

        tmp_img = load_image_cv(img_list[0])

        original_h, original_w, _ = tmp_img.shape
        # keep shorter size less than self.ssize and get the suitable size
        if original_h < original_w:
            if original_h < self.sszie:
                out_h = get_suit_size(original_h)
                out_w = get_suit_size(original_w)
            else:
                out_h = self.sszie
                out_w = get_suit_size(int(original_w * out_h / original_h))
                out_w = get_suit_size(out_w)
        else:
            if original_w < self.sszie:
                out_h = get_suit_size(original_h)
                out_w = get_suit_size(original_w)
            else:
                out_w = self.sszie
                out_h = get_suit_size(int(original_h * out_w / original_w))

        ann_frames = {}
        basename_to_save = list()
        for obj_idx, obj_gt in objs.items():
            obj_idx = int(obj_idx)
            basename_to_save += obj_gt['frames']
            # first frame when this object appear
            frame_idx = basename_list.index(obj_gt['frames'][0])
            obj_st[obj_idx] = frame_idx
            obj_vis[frame_idx:, obj_idx] = 1
            if frame_idx in ann_frames.keys():
                ann_frames[frame_idx]['ids'] += [obj_idx]
            else:
                ann_frames[frame_idx] = {}
                ann_frames[frame_idx]['ids'] = [obj_idx]
                mask_path = os.path.join(mask_dir, obj_gt['frames'][0] + '.png')
                ann_frames[frame_idx]['pth'] = mask_path

        ann_frames = dict(sorted(ann_frames.items(), key=lambda x: x[0]))
        # debug
        # print(ann_frames)

        basename_to_save = sorted(list(set(basename_to_save)))
        # construct onehot masks
        init_masks = {}
        obj_idx_list = [0]

        for frame_id, info in ann_frames.items():
            mask_raw = load_anno_np(info['pth'])  # H x W x 1
            mask_raw = np.squeeze(mask_raw, axis=2)  # H x W
            mask_raw = torch.from_numpy(mask_raw.astype(np.uint8))
            nobjs = len(info['ids'])
            masks = torch.zeros((1, nobjs+1, original_h, original_w))
            current_idx = 1
            masks[0, 0, mask_raw == 0] = 1          # BG
            for obj_id in info['ids']:
                obj_idx_list.append(obj_id)
                masks[0, current_idx, mask_raw == obj_id] = 1
                current_idx += 1

            init_masks[frame_id] = masks
        # debug
        # print(obj_idx_map)
        # first mask
        first_mask_raw = load_anno_np(ann_frames[0]['pth'])  # Ho x Wo x 1
        first_mask_raw = np.squeeze(first_mask_raw, axis=2)      # Ho x Wo
        first_mask_raw = torch.from_numpy(first_mask_raw.astype(np.uint8))

        imgs_np = [load_image_cv(img_list[i], size=(out_h, out_w)) for i in range(video_len)]
        images = torch.stack(self.to_tensor(imgs_np))
        obj_idx_np = np.array(obj_idx_list)
        obj_idx_ten = torch.from_numpy(obj_idx_np)

        info = {
            'name': video_name,
            'num_frames': video_len,
            'obj_vis': obj_vis,
            'obj_st': obj_st,
            'obj_idx_ten': obj_idx_ten,
            'basename_list': basename_list,
            'basename_to_save': basename_to_save,
            'original_size': (original_h, original_w),
            'obj_n': obj_n
        }

        return {'images': images,
                'first_mask': first_mask_raw,
                'init_masks': init_masks,
                'info': info,
                }


if __name__ == '__main__':
    pass
