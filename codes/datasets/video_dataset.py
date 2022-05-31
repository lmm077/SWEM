import os
import math
import numpy as np

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from .data_transform import TPS
from .data_utils import im_mean, reseed, load_image_pil, load_ann_pil, my_print


class FrameSkipper(object):
    def __init__(self, max_skip=25, max_iter=60000):
        self.max_skip = max_skip
        self.final_skip = max(1, self.max_skip * 2//5)
        self.skip_stages = [int(max_iter * 0.1), int(max_iter * 0.8), int(max_iter * 0.9)]
        self.skip_iters = self.get_skip_iters()

    def get_skip_iters(self):
        skip_interval = self.skip_stages[0] / self.max_skip
        skip_iters = [math.ceil(skip_interval * skip) for skip in range(1, self.max_skip + 1)]
        skip_interval = (self.skip_stages[2] - self.skip_stages[1]) / (self.max_skip - self.final_skip)
        skip_iters += [math.ceil(self.skip_stages[1] + skip_interval * skip) for skip in range(1, self.max_skip - self.final_skip + 1)]
        return skip_iters

    def __call__(self, cur_iter):
        if cur_iter <= self.skip_stages[1]:
            skip = int(min((self.max_skip * cur_iter) // self.skip_stages[0], self.max_skip))
        else:
            inter_skip = self.max_skip - self.final_skip
            inter_iter = cur_iter - self.skip_stages[1]
            inter_stage = self.skip_stages[2] - self.skip_stages[1]
            skip = int(max(self.max_skip - (inter_skip * inter_iter) // inter_stage, self.final_skip))

        return skip

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += f'max skip={self.max_skip}'
        format_string += f', final skip={self.final_skip}'
        format_string += f', skip stages={self.skip_stages})'
        format_string += f', skip iters={self.skip_iters})'
        return format_string


class VIDDEODataset(Dataset):
    """
    Works for DAVIS/YouTubeVOS/BL30K training
    For each sequence:
    - Pick three frames
    - Pick two objects
    - Apply some random transforms that are the same for all frames
    - Apply random transform to each of the frame
    - The distance between frames is controlled
    """

    def __init__(self, data_name, config_data, logger, rank, max_iter=60000, is_bl=False):
        assert (data_name in ['DAVIS16', 'DAVIS17', 'YTVOS18', 'YTVOS19'])
        self.cur_path = config_data.PATH
        self.data_name = data_name
        self.seq_len = config_data.NUM_SAMPLE_PER_SEQ
        self.max_nobj = config_data.MAX_NUM_OBJS
        my_print(f'=================================== constructing dataset [{data_name}] with sequence len '
                 f'{self.seq_len}, max nobj {self.max_nobj} ===================================',
                 logger, rank)

        subset_file = None
        if data_name == 'DAVIS16':
            config_data_info = config_data.INFO.DAVIS16
        elif data_name == 'DAVIS17':
            config_data_info = config_data.INFO.DAVIS17
            subset_file = os.path.join(config_data.PATH, 'ImageSets/davis_subset.txt')
        elif data_name == 'YTVOS18':
            config_data_info = config_data.INFO.YTVOS18
            subset_file = os.path.join(config_data.PATH, 'ImageSets/yv_subset.txt')
        elif data_name == 'YTVOS19':
            config_data_info = config_data.INFO.YTVOS19
            subset_file = os.path.join(config_data.PATH, 'ImageSets/yv_subset.txt')
        else:
            raise ValueError(f'Dataset {data_name} is not supported yet.')

        if subset_file is not None:
            with open(subset_file, mode='r') as f:
                subset = set(f.read().splitlines())
        else:
            subset = None
        self.root = config_data_info['root_path']
        if 'DAVIS' in data_name:
            self.im_root = os.path.join(self.root, 'JPEGImages', '480p')
            self.gt_root = os.path.join(self.root, 'Annotations', '480p')
            self.load_size = None
        else:
            # YTVOS18 or 19
            self.load_size = config_data.VID_LOAD_SIZE
            if self.load_size == 480:
                self.im_root = os.path.join(self.root, 'train_480p', 'JPEGImages')
                self.gt_root = os.path.join(self.root, 'train_480p', 'Annotations')
                self.load_size = None
            else:
                self.im_root = os.path.join(self.root, 'train', 'JPEGImages')
                self.gt_root = os.path.join(self.root, 'train', 'Annotations')

        self.crop_size = config_data.VID_CROP_SIZE
        my_print(f' ========= basic information =========', logger, rank)
        my_print(f'Set load size of the short edge as {self.load_size}, crop size {self.crop_size}', logger, rank)
        my_print(f'Set frames root as {self.im_root}', logger, rank)
        my_print(f'Set annotations root as {self.gt_root}', logger, rank)

        self.cur_skip = 0
        self.max_iter = max_iter
        self.max_jump = config_data_info['max_skip']
        self.samples_per_vid = config_data_info['samples_per_video']
        self.skipper = FrameSkipper(self.max_jump, max_iter)
        my_print(f' ========= sample information =========', logger, rank)
        my_print(f'Set frame skipper as {self.skipper}', logger, rank)
        my_print(f'Set samples per-video as {self.samples_per_vid}', logger, rank)

        self.is_bl = is_bl

        self.videos = []
        self.frames = {}

        vid_list = sorted(os.listdir(self.im_root))
        # Pre-filtering
        for vid in vid_list:
            if subset is not None:
                if vid not in subset:
                    continue
            frames = sorted(os.listdir(os.path.join(self.im_root, vid)))
            if len(frames) < 3:
                continue
            self.frames[vid] = frames
            self.videos.append(vid)
        my_print(f' ========= video filtering =========', logger, rank)
        my_print(f'{len(self.videos)} out of {len(vid_list)} videos accepted in', logger, rank)

        my_print(f' ========= transforms =========', logger, rank)
        # These set of transform is the same for im/gt pairs, but different among the 3 sampled frames
        self.pair_im_lone_transform = transforms.Compose([
            transforms.ColorJitter(0.01, 0.01, 0.01, 0),
        ])
        my_print(f'Pair im lone ColorJitter (0.01, 0.01, 0.01, 0)', logger, rank)
        self.pair_im_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=15, shear=10, interpolation=InterpolationMode.BICUBIC, fill=im_mean),
        ])
        my_print(f'Pair im dual RandomAffine(degrees=15, shear=10, interpolation=InterpolationMode.BICUBIC, fill=im_mean)', logger, rank)
        self.pair_gt_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=15, shear=10, interpolation=InterpolationMode.NEAREST, fill=0),
        ])
        my_print(f'Pair gt dual RandomAffine(degrees=15, shear=10, interpolation=InterpolationMode.NEAREST, fill=0)', logger, rank)
        # These transform are the same for all pairs in the sampled sequence
        self.all_im_lone_transform = transforms.Compose([
            transforms.ColorJitter(0.1, 0.03, 0.03, 0),
            transforms.RandomGrayscale(0.05),
        ])
        my_print(f'All im lone ColorJitter(0.1, 0.03, 0.03, 0), RandomGrayscale(0.05)', logger, rank)
        if self.is_bl:
            # Use a different cropping scheme for the blender dataset because the image size is different
            self.all_im_dual_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(self.crop_size, scale=(0.25, 1.00), interpolation=InterpolationMode.BICUBIC)
            ])

            self.all_gt_dual_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(self.crop_size, scale=(0.25, 1.00), interpolation=InterpolationMode.NEAREST)
            ])
            my_print(f'All im and gt dual RandomHorizontalFlip() and  RandomResizedCrop ({self.crop_size}, scale=(0.25, 1.0))', logger, rank)
        else:
            self.all_im_dual_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(self.crop_size, scale=(0.36, 1.00), interpolation=InterpolationMode.BICUBIC)
            ])

            self.all_gt_dual_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(self.crop_size, scale=(0.36, 1.00), interpolation=InterpolationMode.NEAREST)
            ])
            my_print(f'All im and gt dual RandomHorizontalFlip() and  RandomResizedCrop ({self.crop_size}, scale=(0.36, 1.0))', logger, rank)
        # Final transform without randomness, convert image into [0, 1]
        self.final_im_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        my_print(f'Final im ToTensor()', logger, rank)
        self.final_gt_transform = transforms.Compose([
            transforms.PILToTensor(),
        ])
        my_print(f'Final gt PILToTensor()', logger, rank)
        self.tps = TPS(margin_rate=0.25, p=1)
        my_print(f'Pair dual {self.tps}', logger, rank)
        my_print(f'=================================== date set [{data_name}] has been constructed =================================== ', logger, rank)

    def set_max_skip(self, cur_iter):
        self.cur_skip = min(self.skipper(cur_iter), self.max_jump)

    def _select(self, left_ids, right_ids, selected_ids, n_frame):
        left_seq_len = len(left_ids)
        right_seq_len = len(right_ids)
        mean_inter = (left_seq_len + right_seq_len - n_frame) // n_frame
        left_seq_len = min(left_seq_len, self.cur_skip + 1, mean_inter + 1)
        right_seq_len = min(right_seq_len, self.cur_skip + 1, mean_inter + 1)
        idx = np.random.randint(-left_seq_len, right_seq_len)
        if idx >= 0:
            selected_idx = right_ids[idx]
            selected_ids.append(selected_idx)
            right_ids = right_ids[idx+1:]
        else:
            selected_idx = left_ids[idx]
            selected_ids.append(selected_idx)
            left_ids = left_ids[:idx]

        if n_frame - 1 <= 0:
            return selected_ids
        else:
            return self._select(left_ids, right_ids, selected_ids, n_frame-1)

    def select_frames(self, frame_ids):
        if self.seq_len > len(frame_ids):
            selected_ids = np.random.choice(frame_ids, size=self.seq_len, replace=True)
            return selected_ids

        idx = np.random.randint(0, len(frame_ids))
        selected_ids = [frame_ids[idx]]
        left_ids = frame_ids[:idx]
        right_ids = frame_ids[idx+1:]
        selected_ids = self._select(left_ids, right_ids, selected_ids, self.seq_len-1)
        return selected_ids

    def __getitem__(self, idx):
        true_idx = idx // self.samples_per_vid
        video = self.videos[true_idx]
        info = {}
        info['dataset'] = self.data_name
        info['name'] = video

        vid_im_path = os.path.join(self.im_root, video)
        vid_gt_path = os.path.join(self.gt_root, video)
        frames = self.frames[video]

        trials = 0
        ids = [i for i in range(len(frames))]
        while trials < 5:
            info['frames'] = []  # Appended with actual frames
            # Don't want to bias towards beginning/end
            '''
            this_max_jump = min(len(frames) - 3, self.cur_skip)
            start_idx = np.random.randint(len(frames) - this_max_jump - 3 + 1)
            f1_idx = start_idx + np.random.randint(this_max_jump + 1) + 1
            f1_idx = min(f1_idx, len(frames) - this_max_jump, len(frames) - 1)

            f2_idx = f1_idx + np.random.randint(this_max_jump + 1) + 1
            f2_idx = min(f2_idx, len(frames) - this_max_jump // 2, len(frames) - 1)

            frames_idx = [start_idx, f1_idx, f2_idx]
            '''
            frames_idx = self.select_frames(ids)
            frames_idx = sorted(frames_idx)
            skips = [frames_idx[i] - frames_idx[i-1] for i in range(1, len(frames_idx))]

            if np.random.rand() < 0.5:
                # Reverse time
                frames_idx = frames_idx[::-1]

            sequence_seed = np.random.randint(2147483647)
            images = []
            masks = []
            for i, f_idx in enumerate(frames_idx):
                
                jpg_name = frames[f_idx][:-4] + '.jpg'
                png_name = frames[f_idx][:-4] + '.png'
                info['frames'].append(jpg_name)

                # all sequence transforms
                reseed(sequence_seed)
                this_im = load_image_pil(os.path.join(vid_im_path, jpg_name), size=self.load_size)
                this_im = self.all_im_dual_transform(this_im)
                this_im = self.all_im_lone_transform(this_im)
                reseed(sequence_seed)
                this_gt = load_ann_pil(os.path.join(vid_gt_path, png_name), size=self.load_size)
                this_gt = self.all_gt_dual_transform(this_gt)

                # pairwise transforms
                pairwise_seed = np.random.randint(2147483647)
                reseed(pairwise_seed)
                this_im = self.pair_im_dual_transform(this_im)
                this_im = self.pair_im_lone_transform(this_im)
                reseed(pairwise_seed)
                this_gt = self.pair_gt_dual_transform(this_gt)

                this_im = self.final_im_transform(this_im)
                this_gt = self.final_gt_transform(this_gt).float()

                images.append(this_im)
                masks.append(this_gt)

            # tps
            images, masks = self.tps(images, masks)

            labels = torch.unique(masks[0].long()).tolist()
            # Remove background
            if 0 in labels: labels.remove(0)
            if 255 in labels: labels.remove(255)

            if self.is_bl:
                # Find large enough labels
                good_lables = []
                for l in labels:
                    pixel_sum = (masks[0] == l).sum()
                    if pixel_sum > 10 * 10:
                        # OK if the object is always this small
                        # Not OK if it is actually much bigger
                        if pixel_sum > 30 * 30:
                            good_lables.append(l)
                        elif max((masks[1] == l).sum(), (masks[2] == l).sum()) < 20 * 20:
                            good_lables.append(l)
                labels = np.array(good_lables, dtype=np.uint8)

            if len(labels) == 0:
                selected_labels = [-1]  # all black if no objects
                trials += 1
                nobj_ = 1
            else:
                nobj_ = min(self.max_nobj, len(labels))
                selected_labels = np.random.choice(labels, nobj_, replace=False)
                break

        images = torch.stack(images, dim=0)                                                         # T, 3, H, W

        # get one-hot masks
        masks = torch.cat(masks, dim=0).long()                                                    # T, H, W
        # print(masks.shape)
        tar_masks = [(masks == selected_labels[obj_idx]).long() for obj_idx in range(nobj_)]
        tar_masks += [torch.zeros_like(tar_masks[0]) for _ in range(nobj_, self.max_nobj)]
        ## [bg, fg]
        fg_masks = torch.stack(tar_masks, dim=1)                              # T, N, H, W
        bg_mask = torch.ones_like(tar_masks[0]) - torch.sum(fg_masks, dim=1)  # T, H, W
        bg_mask[bg_mask != 1] = 0
        masks = torch.cat([bg_mask.unsqueeze(1), fg_masks], dim=1)            # T, N+1, H, W

        # get selector
        selector = [1] * (nobj_ + 1) + [0] * (self.max_nobj - nobj_)               # N+1
        selector = torch.FloatTensor(selector)

        # skips
        skips = torch.tensor(skips).float().mean() - 1

        info['size'] = images.shape[-3:]

        data = {
            'images': images,
            'masks': masks,
            'valid_obj': selector,
            'skips': skips,
            'info': info
        }

        return data

    def __len__(self):
        return len(self.videos) * self.samples_per_vid
