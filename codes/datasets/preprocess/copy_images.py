import sys
import os
from os import path
from glob import glob
from shutil import copyfile
from progressbar import progressbar
from multiprocessing import Pool


def copy_imgs(inputs):
    src_path, dst_path = inputs
    copyfile(src_path, dst_path)


def copy_all(in_path, out_path, folders):
    print('Processing images')
    data_dir = os.path.join(in_path, 'JPEGImages')
    os.makedirs(path.join(out_path, 'JPEGImages'), exist_ok=True)
    src_img_list = []
    tar_img_list = []
    for folder_name in folders:
        img_list = sorted(glob(os.path.join(data_dir, folder_name, '*.jpg'))) + sorted(glob(os.path.join(data_dir, folder_name, '*.png')))
        src_img_list += img_list
        for img_dir in img_list:
            tar_dir = os.path.join(out_path, 'JPEGImages', folder_name, os.path.basename(img_dir))
            tar_img_list.append(tar_dir)
        os.makedirs(path.join(out_path, 'JPEGImages', folder_name), exist_ok=True)
    all_imgs = [(src_img_dir, tar_img_dir) for src_img_dir, tar_img_dir in zip(src_img_list, tar_img_list)]

    pool = Pool(processes=8)
    for _ in progressbar(pool.imap_unordered(copy_imgs, all_imgs), max_value=len(all_imgs)):
        pass

    print('Processing annotations')
    data_dir = os.path.join(in_path, 'Annotations')
    os.makedirs(path.join(out_path, 'Annotations'), exist_ok=True)
    src_ann_list = []
    tar_ann_list = []
    for folder_name in folders:
        ann_list = sorted(glob(os.path.join(data_dir, folder_name, '*.png')))
        src_ann_list += ann_list
        for ann_dir in ann_list:
            tar_dir = os.path.join(out_path, 'Annotations', folder_name, os.path.basename(ann_dir))
            tar_ann_list.append(tar_dir)
        os.makedirs(path.join(out_path, 'Annotations', folder_name), exist_ok=True)
    all_anns = [(src_ann_dir, tar_ann_dir) for src_ann_dir, tar_ann_dir in zip(src_ann_list, tar_ann_list)]

    pool = Pool(processes=8)
    for _ in progressbar(pool.imap_unordered(copy_imgs, all_anns), max_value=len(all_anns)):
        pass


if __name__ == '__main__':
    in_path = "/data/lin-zh/datasets/VOS/PreTrain/"
    out_path = "/data/lin-zh/datasets/VOS/STCN_PreTrain/"
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)
    copy_all(in_path, out_path, folders=['BIG_small', 'HRSOD_small', 'FSS', 'ECSSD', 'DUTS'])

    print('Done.')