import random
from PIL import Image
import cv2
import numpy as np

import torch

im_mean = (124, 116, 104)


def reseed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def my_print(info, logger=None, rank=0):
    if rank <= 0:
        print(info)
        if logger is not None:
            logger.info(info)


def load_image_pil(path, size=None):
    img = Image.open(path)
    img.load()  # Very important for loading large image
    img = img.convert('RGB')
    if size is not None:
        if isinstance(size, list) or isinstance(size, tuple):
            img = img.resize((size[1], size[0]), Image.BICUBIC)
        elif isinstance(size, int):
            assert (size > 0)
            w, h = img.size
            if h < w:
                nh = size
                nw = nh * w // h
            else:
                nw = size
                nh = nw * h // w
            img = img.resize((nw, nh), Image.BICUBIC)

    return img  # [0-255]


def load_ann_pil(path, size=None):
    ann = Image.open(path)
    ann.load()  # Very important for loading large image
    ann = ann.convert('P')
    if size is not None:
        if isinstance(size, list) or isinstance(size, tuple):
            ann = ann.resize((size[1], size[0]), Image.NEAREST)
        elif isinstance(size, int):
            w, h = ann.size
            if h < w:
                nh = size
                nw = nh * w // h
            else:
                nw = size
                nh = nw * h // w
            ann = ann.resize((nw, nh), Image.NEAREST)

    return ann


def get_obj_ids(ann_path, threshold, size=None):
    """
       return the object ids which area is bigger than the thresholds
    """
    ann = load_ann_pil(ann_path, size)
    tenAnn = torch.from_numpy(np.array(ann, np.uint8, copy=False)).long().view(1, ann.size[1], ann.size[0])
    values = (tenAnn.view(-1).bincount() > threshold).nonzero().view(-1).tolist()
    # remove background
    if 0 in values: values.remove(0)
    if 255 in values: values.remove(255)
    return values


def pil_to_long_tensor(pic):
    if isinstance(pic, np.ndarray):
        # handle numpy array
        label = torch.from_numpy(pic).long()
    elif pic.mode == '1':
        label = torch.from_numpy(np.array(pic, np.uint8, copy=False)).long().view(1, pic.size[1], pic.size[0])
    else:
        label = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        if pic.mode == 'LA': # Hack to remove alpha channel if it exists
            label = label.view(pic.size[1], pic.size[0], 2)
            label = label.transpose(0, 1).transpose(0, 2).contiguous().long()[0]
            label = label.view(1, label.size(0), label.size(1))
        else:
            label = label.view(pic.size[1], pic.size[0], -1)
            label = label.transpose(0, 1).transpose(0, 2).contiguous().long()
    return label


# load images with opencv
def load_image_cv(path, size=None):
    img = cv2.imread(path)
    if img is None:
        print(f'image {path} is not existed')
    assert (img is not None)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img, dtype=np.float32) / 255.0
    if size is not None:
        if isinstance(size, list) or isinstance(size, tuple):
            img = cv2.resize(img, (size[1], size[0]), cv2.INTER_CUBIC)
        elif isinstance(size, int):
            assert (size > 0)
            h, w, _ = img.shape
            if h < w:
                nh = size
                nw = nh * w // h
            else:
                nw = size
                nh = nw * h // w
            img = cv2.resize(img, (nw, nh), cv2.INTER_CUBIC)
    return img


# load masks
def load_anno_np(path, size=None):
    ann = Image.open(path)
    ann.load()  # Very important for loading large image
    if size is not None:
        if isinstance(size, list) or isinstance(size, tuple):
            ann = ann.resize((size[1], size[0]), Image.NEAREST)
        elif isinstance(size, int):
            w, h = ann.size
            if h < w:
                nh = size
                nw = nh * w // h
            else:
                nw = size
                nh = nw * h // w
            ann = ann.resize((nw, nh), Image.NEAREST)

    tenAnn = pil_to_long_tensor(ann)
    ann = tenAnn.numpy().transpose([1, 2, 0]) # H x W x 1
    return ann


def to_onehot_tensor(mask, max_obj_n, obj_list=None, shuffle=True, valid_shuffle=True):
    """
        Args:
            mask (Mask in Numpy, HxW): Mask to be converted.
            max_obj_n (float): Maximum number of the objects
        Returns:
            Tensor: Converted mask in onehot format.
        """
    new_mask = np.zeros((max_obj_n, *mask.shape), np.uint8)
    valid_obj = None

    if obj_list is None:
        valid_obj = [1]                 # always valid for background
        obj_list = list()
        # existing objects
        for i in range(1, mask.max() + 1):
            tmp = (mask == i).astype(np.uint8)
            if tmp.max() > 0:
                obj_list.append(i)
                valid_obj.append(1)
        if valid_shuffle:
            random.shuffle(obj_list)        # always shuffle
        # adding non-exist object to align the max_obj_n
        n_remain_objects = max_obj_n - 1 - len(obj_list)
        if n_remain_objects > 0:
            pseudo_id = 999
            for i in range(n_remain_objects):
                obj_list.append(pseudo_id)
                valid_obj.append(0)

        # shuffle
        if shuffle:
            random.shuffle(obj_list)
            valid_obj = None
        else:
            valid_obj = np.array(valid_obj[:max_obj_n])
        obj_list = obj_list[:max_obj_n - 1]

    # foreground
    for i, obj_id in enumerate(obj_list):
        new_mask[i + 1] = (mask == obj_id).astype(np.uint8)

    # background
    new_mask[0] = 1 - np.sum(new_mask, axis=0)

    return new_mask, obj_list, valid_obj