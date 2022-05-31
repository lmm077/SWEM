import os
import numpy as np
from glob import glob

import torch
from torch.utils import data

from .data_transform import ToTensor
from .data_utils import load_image_cv, load_anno_np, to_onehot_tensor


class DAVIS_Test(data.Dataset):

    def __init__(self, root, img_set='2016/val.txt', single_obj=False):
        self.root = root
        self.single_obj = single_obj
        dataset_path = os.path.join(root, 'ImageSets', img_set)
        self.dataset_list = list()

        with open(os.path.join(dataset_path), 'r') as lines:
            for line in lines:
                dataset_name = line.strip()
                if len(dataset_name) > 0:
                    self.dataset_list.append(dataset_name)

        self.to_tensor = ToTensor()  # convert img_np list to tensor list

    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, idx):
        video_name = self.dataset_list[idx]

        img_dir = os.path.join(self.root, 'JPEGImages', '480p', video_name)
        mask_dir = os.path.join(self.root, 'Annotations', '480p', video_name)

        img_list = sorted(glob(os.path.join(img_dir, '*.jpg')))
        mask_list = sorted(glob(os.path.join(mask_dir, '*.png')))

        first_mask_np = load_anno_np(mask_list[0])  # H x W x 1
        first_mask_np = np.squeeze(first_mask_np, axis=2)

        if self.single_obj:
            first_mask_np[first_mask_np > 1] = 1

        obj_n = first_mask_np.max() + 1
        video_len = len(img_list)
        mask_np, _, _ = to_onehot_tensor(first_mask_np, obj_n, shuffle=False, valid_shuffle=False)  # obj_n x H x W
        masks = torch.from_numpy(mask_np).unsqueeze(0).long()                                  # 1 x obj_n x H x W

        imgs_np = [load_image_cv(img_list[i]) for i in range(video_len)]                   # list of H x W x C
        images = torch.stack(self.to_tensor(imgs_np))                                          # T x C x H x W

        info = {
            'name': video_name,
            'num_frames': video_len,
            'obj_n': obj_n,
        }

        return {'images': images, 
                'masks': masks,
                'info': info
                }


if __name__ == '__main__':
    pass
