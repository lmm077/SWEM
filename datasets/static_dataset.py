import os
import random
import math

import cv2
from PIL import Image
import numpy as np
from glob import glob

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from .data_transform import TPS
from .data_utils import im_mean, reseed, load_image_pil, load_ann_pil, my_print


def get_bbox(msk):
    if msk.sum() > 0:
        coor = np.nonzero(msk)
        hmin = coor[0][0]
        hmax = coor[0][-1]
        coor[1].sort()  #
        wmin = coor[1][0]
        wmax = coor[1][-1]
        bbox = [hmin, wmin, hmax+1, wmax+1]
        #width = wmax - wmin
        #height = hmax - hmin
        #wcentor = (wmax + wmin) // 2
        #hcentor = (hmax + hmin) // 2
        return bbox
    else:
        return None


def crop(img, msk, bbox):
    c_img = img[bbox[0]:bbox[2], bbox[1]:bbox[3]]
    c_msk = msk[bbox[0]:bbox[2], bbox[1]:bbox[3]]
    return c_img, c_msk


def random_resize(img, msk, scale=(0.16, 0.81), ratio=(3./4., 4./3.)):
    h, w = img.shape[:2]
    target_area = random.uniform(*scale) * (h*w)
    log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
    aspect_ratio = math.exp(random.uniform(*log_ratio))
    h = int(round(math.sqrt(target_area / aspect_ratio)))
    w = int(round(math.sqrt(target_area * aspect_ratio)))

    rr_img = cv2.resize(img, dsize=(w, h), interpolation=cv2.INTER_CUBIC)
    rr_msk = cv2.resize(msk, dsize=(w, h), interpolation=cv2.INTER_NEAREST)

    return rr_img, rr_msk


def place_object(img, msk, tar_size):
    #print(f'image shape {img.shape}, mask shape {msk.shape}')
    msk = msk[:, :, np.newaxis]
    h, w, c = tar_size
    p_img = np.zeros((h, w, c))
    p_msk = np.zeros((h, w, 1))
    src_h, src_w = img.shape[:2]
    center_x = random.randint(src_w//2, max(w - src_w//2, src_w//2))
    center_y = random.randint(src_h//2, max(h - src_h//2, src_h//2))
    top_left_x = center_x - src_w//2
    min_x = max(0, src_w//2 - center_x)
    top_left_y = center_y - src_h//2
    min_y = max(0, src_h//2 - center_y)
    down_right_x = min(w, top_left_x + src_w)
    down_right_y = min(h, top_left_y + src_h)
    top_left_x = max(0, top_left_x)
    top_left_y = max(0, top_left_y)
    tar_h = down_right_y - top_left_y
    tar_w = down_right_x - top_left_x
    p_img[top_left_y:down_right_y, top_left_x:down_right_x] = img[min_y:min_y+tar_h, min_x:min_x+tar_w] * msk[min_y:min_y+tar_h, min_x:min_x+tar_w]
    p_msk[top_left_y:down_right_y, top_left_x:down_right_x] = msk[min_y:min_y+tar_h, min_x:min_x+tar_w]

    return p_img, p_msk


def synthesis_frames(imgs, msks, nframe=3):
    """
    Args:
        imgs: list of n images
        msks: list of n masks
    Returns:
        syn_imgs: synthesis frames with the same background as the 0th image and the same foreground objects as all images
        syn_msks: corresponding mask for the syn_imgs
    """
    if len(imgs) == 1:
        syn_imgs = [imgs[0] for _ in range(nframe)]
        syn_msks = [msks[0] for _ in range(nframe)]
        return syn_imgs, syn_msks

    imgs = [np.array(img).astype(np.float32) for img in imgs]
    msks = [np.array(msk)[:, :, np.newaxis] for msk in msks]

    h, w, c = imgs[0].shape
    # 0. get bboxes
    bboxes = [get_bbox(msk) for msk in msks]
    # 1. crop objects
    c_imgs, c_msks = [], []
    for img, msk, bbox in zip(imgs, msks, bboxes):
        if bbox is not None:
            c_img, c_msk = crop(img, msk, bbox)
            #print(f'image shape after crop {c_img.shape}')
            #print(f'mask shape after crop {c_msk.shape}')
            c_imgs.append(c_img)
            c_msks.append(c_msk)
    # 2. get ids
    ids = [i for i in range(1, len(c_imgs)+2)]
    random.shuffle(ids)
    # 3. synthesis frames
    syn_imgs, syn_msks = [], []
    for _ in range(nframe):
        # 3.0 random resize objects
        rr_imgs, rr_msks = [], []
        for img, msk in zip(c_imgs, c_msks):
            rr_img, rr_msk = random_resize(img, msk)
            rr_imgs.append(rr_img)
            rr_msks.append(rr_msk)
            #print(f'image shape after resize {rr_img.shape}')
            #print(f'mask shape after resize {rr_msk.shape}')

        # place objects in the target frame
        #p_imgs, p_msks = [imgs[0] * msks[0]], [msks[0]]
        p_imgs, p_msks = [], []
        for img, msk in zip(rr_imgs, rr_msks):
            p_img, p_msk = place_object(img, msk, (h, w, c))
            p_imgs.append(p_img)
            p_msks.append(p_msk)

        # synthesis frame
        mean_fg = np.sum(imgs[0] * msks[0], axis=(0, 1), keepdims=True) / (np.sum(msks[0], axis=(0, 1), keepdims=True) + 1e-8)
        syn_img = imgs[0] * (1 - msks[0]) + mean_fg * msks[0]   # BG
        syn_msk = np.zeros_like(msks[0])
        # random order
        orders = [i for i in range(len(p_imgs))]
        random.shuffle(orders)
        for i in orders:
            syn_img = syn_img * (1 - p_msks[i]) + p_imgs[i] * p_msks[i]
            syn_msk[p_msks[i] == 1] = ids[i]
        syn_imgs.append(Image.fromarray(syn_img.astype(np.uint8)))
        syn_msks.append(Image.fromarray(syn_msk[:, :, 0]).convert('P'))

    return syn_imgs, syn_msks


class StaticTransformDataset(Dataset):
    """
    Generate pseudo VOS data by applying random transforms on static images.
    Single-object only.
    """
    def __init__(self, config_data, logger, rank):
        self.root = self.root = config_data.INFO.PRETRAIN['root_path']
        self.seq_len = config_data.NUM_SAMPLE_PER_SEQ
        self.max_nobj = config_data.MAX_NUM_OBJS
        my_print(
            f'=================================== constructing dataset [Image] with sequence len {self.seq_len}, '
            f'max nobj {self.max_nobj} ===================================',
            logger, rank)

        self.img_list, self.msk_list = list(), list()
        self.real_img_list, self.real_msk_list = list(), list()
        dataset_list = config_data.PRETRAIN_SET
        dataset_ratio = config_data.PRETRAIN_SET_RATIO
        assert (len(dataset_list) == len(dataset_ratio))
        for dataset_name, ratio in zip(dataset_list, dataset_ratio):
            img_dir = os.path.join(self.root, 'JPEGImages', dataset_name)
            mask_dir = os.path.join(self.root, 'Annotations', dataset_name)

            img_list = sorted(glob(os.path.join(img_dir, '*.jpg'))) + sorted(glob(os.path.join(img_dir, '*.png')))
            msk_list = sorted(glob(os.path.join(mask_dir, '*.png')))

            assert len(img_list) == len(msk_list)

            for img_name, mask_name in zip(img_list, msk_list):
                img_base = os.path.basename(img_name)
                msk_base = os.path.basename(mask_name)
                if img_base[:-4] != msk_base[:-4]:
                    print(f'Unmatched name img {img_name}, msk {mask_name}')
                assert (img_base[:-4] == msk_base[:-4])

            self.img_list += (img_list * ratio)
            self.msk_list += (msk_list * ratio)
            self.real_img_list += img_list
            self.real_msk_list += msk_list

        self.img_index_map = {}
        for i, img_name in enumerate(self.real_img_list):
            self.img_index_map[img_name] = i
        self.real_len = len(self.real_img_list)

        my_print(f'Creating Image Dataset by {dataset_list} with ratios {dataset_ratio}, total_have {len(self.img_list)} samples.', logger, rank)

        # These set of transform is the same for im/gt pairs, but different among the 3 sampled frames
        self.pair_im_lone_transform = transforms.Compose([
            transforms.ColorJitter(0.1, 0.05, 0.05, 0),  # No hue change here as that's not realistic
        ])

        self.pair_im_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=20, scale=(0.9, 1.1), shear=10, interpolation=InterpolationMode.BICUBIC, fill=im_mean),
            transforms.Resize(384, InterpolationMode.BICUBIC),
            transforms.RandomCrop((384, 384), pad_if_needed=True, fill=im_mean),
        ])

        self.pair_gt_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=20, scale=(0.9, 1.1), shear=10, interpolation=InterpolationMode.BICUBIC, fill=0),
            transforms.Resize(384, InterpolationMode.NEAREST),
            transforms.RandomCrop((384, 384), pad_if_needed=True, fill=0),
        ])

        # These transform are the same for all pairs in the sampled sequence
        self.all_im_lone_transform = transforms.Compose([
            transforms.ColorJitter(0.1, 0.05, 0.05, 0.05),
            transforms.RandomGrayscale(0.05),
        ])

        self.all_im_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=0, scale=(0.8, 1.5), fill=im_mean),
            transforms.RandomHorizontalFlip(),
        ])

        self.all_gt_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=0, scale=(0.8, 1.5), fill=0),
            transforms.RandomHorizontalFlip(),
        ])

        # Final transform without randomness
        self.final_im_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.final_gt_transform = transforms.Compose([
            transforms.PILToTensor(),
        ])

        self.tps = TPS(margin_rate=0.3, p=1)
        my_print(f'TPS {self.tps}', logger, rank)

    def __getitem__(self, idx):
        im = load_image_pil(self.img_list[idx])
        gt = load_ann_pil(self.msk_list[idx])
        ims, gts = [im], [gt]
        if self.max_nobj == 1:
            ims, gts = synthesis_frames(ims, gts, nframe=self.seq_len)
        else:
            res_ids = np.random.choice(self.real_len-1, size=self.max_nobj-1, replace=False)
            res_ids = list(res_ids)
            cur_id = self.img_index_map[self.img_list[idx]]
            real_ids = [(res_id + cur_id) % self.real_len for res_id in res_ids]
            ims += [load_image_pil(self.real_img_list[real_id]) for real_id in real_ids]
            gts += [load_ann_pil(self.real_msk_list[real_id]) for real_id in real_ids]
            ims, gts = synthesis_frames(ims, gts, nframe=self.seq_len)

        sequence_seed = np.random.randint(2147483647)

        images = []
        masks = []
        for im, gt in zip(ims, gts):
            reseed(sequence_seed)
            this_im = self.all_im_dual_transform(im)
            this_im = self.all_im_lone_transform(this_im)
            reseed(sequence_seed)
            this_gt = self.all_gt_dual_transform(gt)

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
        label_ids = torch.unique(masks[0].long(), sorted=True).tolist()
        if 0 in label_ids: label_ids.remove(0)
        if 255 in label_ids: label_ids.remove(255)
        if len(label_ids) > self.max_nobj:
            label_ids = np.random.choice(label_ids, self.max_nobj, replace=False)
        label_ids = [0] + label_ids

        images = torch.stack(images, 0)      # T, C, H, W
        masks = torch.stack(masks, 0)        # T, 1, H, W
        tar_masks = [(masks == label_id).long() for label_id in label_ids]
        tar_masks += [torch.zeros_like(tar_masks[0]) for _ in range(len(label_ids)-1, self.max_nobj)]
        masks = torch.cat(tar_masks, dim=1)  # T, N+1, H, W
        selector = [1] * len(label_ids) + [0] * (self.max_nobj - len(label_ids) + 1)  # N+1
        selector = torch.FloatTensor(selector)

        info = {
            'name': self.img_list[idx],
            'frame': [i for i in range(3)],
            'size': images.size()[-3:]
        }

        data = {
            'images': images,
            'masks': masks,
            'valid_obj': selector,
            'info': info
        }

        return data

    def __len__(self):
        return len(self.img_list)


if __name__ == "__main__":
    # debug
    from scipy.ndimage.morphology import binary_dilation


    def add_overlay(img, mask, colors, alpha=0.7, cscale=1):
        mask = np.array(mask)
        # print(mask.shape)
        # print(np.max(mask))

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


    def save_overlay(img, mask, overlay_path, colors=(255, 0, 0), alpha=0.4, cscale=1):

        img = np.array(img).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        img_overlay = add_overlay(img, mask, colors, alpha, cscale)
        cv2.imwrite(overlay_path, img_overlay)


    img_folder = "/home/linzhihui/labs/HumanFace/Video/VOS/codes/debug/syn_data/images/"
    msk_folder = "/home/linzhihui/labs/HumanFace/Video/VOS/codes/debug/syn_data/masks/"
    tar_folder = "/home/linzhihui/labs/HumanFace/Video/VOS/codes/debug/syn_data/results/"
    img_list = os.listdir(img_folder)
    imgs, msks = [], []
    for img_name in img_list:
        img_path = os.path.join(img_folder, img_name)
        imgs.append(load_image_pil(img_path))
        msk_path = os.path.join(msk_folder, img_name[:-4]+'.png')
        msks.append(load_ann_pil(msk_path))
    palette = msks[0].getpalette()

    syn_imgs, syn_msks = synthesis_frames(imgs, msks, nframe=3)
    i = 0
    for img, msk in zip(syn_imgs, syn_msks):
        tar_img_path = os.path.join(tar_folder, f'{i}.jpg')
        img.save(tar_img_path, mode='RGB')
        tar_msk_path = os.path.join(tar_folder, f'{i}.png')
        msk.save(tar_msk_path, mode='P')
        overlay_path = os.path.join(tar_folder, f'{i}_overlay.jpg')
        save_overlay(np.array(img), msk, overlay_path, palette)
        i += 1


