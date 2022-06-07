import os
import sys
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
from PIL import Image
from .davis import DAVIS
from .metrics import db_eval_boundary, db_eval_iou
from . import utils
from .results import Results
from scipy.optimize import linear_sum_assignment

import torch
import time
import math

class DAVISEvaluation(object):
    def __init__(self, davis_root, year, task, gt_set, store_results=False, res_root=None, sequences='all', codalab=False):
        """
        Class to evaluate DAVIS sequences from a certain set and for a certain task
        :param davis_root: Path to the DAVIS folder that contains JPEGImages, Annotations, etc. folders.
        :param task: Task to compute the evaluation, chose between semi-supervised or unsupervised.
        :param gt_set: Set to compute the evaluation
        :param sequences: Sequences to consider for the evaluation, 'all' to use all the sequences in a set.
        """
        self.davis_root = davis_root
        self.task = task
        self.dataset = DAVIS(root=davis_root, year=year, task=task, subset=gt_set, sequences=sequences, codalab=codalab)
        
        # Store pred masks
        self.store_results = store_results
        if res_root is None:
            self.res_root = davis_root + '_Pred'
        else:
            self.res_root = res_root
        if self.store_results:
            if not os.path.exists(self.res_root):
                os.makedirs(self.res_root)
            if not os.path.exists('{}/{}'.format(self.res_root, 'overlay')):
                os.mkdir('{}/{}'.format(self.res_root, 'overlay'))
            if not os.path.exists('{}/{}'.format(self.res_root, 'results')):
                os.mkdir('{}/{}'.format(self.res_root, 'results'))

    @staticmethod
    def _evaluate_semisupervised(all_gt_masks, all_res_masks, all_void_masks, metric):
        if all_res_masks.shape[0] > all_gt_masks.shape[0]:
            sys.stdout.write("\nIn your PNG files there is an index higher than the number of objects in the sequence!")
            sys.exit()
        elif all_res_masks.shape[0] < all_gt_masks.shape[0]:
            zero_padding = np.zeros((all_gt_masks.shape[0] - all_res_masks.shape[0], *all_res_masks.shape[1:]))
            all_res_masks = np.concatenate([all_res_masks, zero_padding], axis=0)
        j_metrics_res, f_metrics_res = np.zeros(all_gt_masks.shape[:2]), np.zeros(all_gt_masks.shape[:2])
        for ii in range(all_gt_masks.shape[0]):
            if 'J' in metric:
                j_metrics_res[ii, :] = db_eval_iou(all_gt_masks[ii, ...], all_res_masks[ii, ...], all_void_masks)
            if 'F' in metric:
                f_metrics_res[ii, :] = db_eval_boundary(all_gt_masks[ii, ...], all_res_masks[ii, ...], all_void_masks)
        return j_metrics_res, f_metrics_res

    @staticmethod
    def _evaluate_unsupervised(all_gt_masks, all_res_masks, all_void_masks, metric, max_n_proposals=20):
        if all_res_masks.shape[0] > max_n_proposals:
            sys.stdout.write(f"\nIn your PNG files there is an index higher than the maximum number ({max_n_proposals}) of proposals allowed!")
            sys.exit()
        elif all_res_masks.shape[0] < all_gt_masks.shape[0]:
            zero_padding = np.zeros((all_gt_masks.shape[0] - all_res_masks.shape[0], *all_res_masks.shape[1:]))
            all_res_masks = np.concatenate([all_res_masks, zero_padding], axis=0)
        j_metrics_res = np.zeros((all_res_masks.shape[0], all_gt_masks.shape[0], all_gt_masks.shape[1]))
        f_metrics_res = np.zeros((all_res_masks.shape[0], all_gt_masks.shape[0], all_gt_masks.shape[1]))
        for ii in range(all_gt_masks.shape[0]):
            for jj in range(all_res_masks.shape[0]):
                if 'J' in metric:
                    j_metrics_res[jj, ii, :] = db_eval_iou(all_gt_masks[ii, ...], all_res_masks[jj, ...], all_void_masks)
                if 'F' in metric:
                    f_metrics_res[jj, ii, :] = db_eval_boundary(all_gt_masks[ii, ...], all_res_masks[jj, ...], all_void_masks)
        if 'J' in metric and 'F' in metric:
            all_metrics = (np.mean(j_metrics_res, axis=2) + np.mean(f_metrics_res, axis=2)) / 2
        else:
            all_metrics = np.mean(j_metrics_res, axis=2) if 'J' in metric else np.mean(f_metrics_res, axis=2)
        row_ind, col_ind = linear_sum_assignment(-all_metrics)
        return j_metrics_res[row_ind, col_ind, :], f_metrics_res[row_ind, col_ind, :]

    ### save mask
    @staticmethod
    def _save_mask(mask, img_path, colors):
        if np.max(mask) > 255:
            raise ValueError('Maximum id pixel value is 255')
        mask_img = Image.fromarray(mask.astype(np.uint8))
        mask_img.putpalette(colors)
        mask_img.save(img_path)

    ### get palette
    @staticmethod
    def _get_palette():
        palette=[]
        for i in range(256):
            palette.extend((i,i,i))

        palette[:3*21]=np.array([
            [0, 0, 0],
            [128, 0, 0],
            [0, 128, 0],
            [128, 128, 0],
            [0, 0, 128],
            [128, 0, 128],
            [0, 128, 128],
            [128, 128, 128],
            [64, 0, 0],
            [192, 0, 0],
            [64, 128, 0],
            [192, 128, 0],
            [64, 0, 128],
            [192, 0, 128],
            [64, 128, 128],
            [192, 128, 128],
            [0, 64, 0],
            [128, 64, 0],
            [0, 192, 0],
            [128, 192, 0],
            [0, 64, 128]
            ], dtype='uint8').flatten()

        return palette

    # save overlay
    @staticmethod
    def _overlay_davis(image, mask, img_path, colors=[255,0,0],cscale=2,alpha=0.7):
        """ Overlay segmentation on top of RGB image. from davis official"""
        # import skimage
        from scipy.ndimage.morphology import binary_erosion, binary_dilation
        colors = np.reshape(colors, (-1, 3))
        colors = np.atleast_2d(colors) * cscale

        im_overlay = image.copy()
        object_ids = np.unique(mask)
        #print(colors.shape)

        for object_id in object_ids[1:]:
            # Overlay color on  binary mask
            #print('obj_id {}'.format(object_id))
            #print(colors[object_id].shape)
            foreground = image*alpha + np.ones(image.shape)*(1-alpha) * np.array(colors[object_id])
            binary_mask = mask == object_id

            # Compose image
            im_overlay[binary_mask] = foreground[binary_mask]

            # countours = skimage.morphology.binary.binary_dilation(binary_mask) - binary_mask
            countours = binary_dilation(binary_mask) ^ binary_mask
            # countours = cv2.dilate(binary_mask, cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))) - binary_mask
            im_overlay[countours,:] = 0
        
        im_overlay.astype(image.dtype)
        im_overlay = Image.fromarray(im_overlay)
        im_overlay.save(img_path)

    ### Mask to Tensor
    @staticmethod
    def _LabelToLongTensor(pic):
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
    
    ### msk tensor to msk dict
    @staticmethod
    def _msk2dict(msk, obj_ids = None):
        # just for inference
        #assert(masks.size(0) == 1)
        masks_dict = {}
        if obj_ids is None:
            possible_obj_ids = msk.unique().tolist()
            if 0 in possible_obj_ids: possible_obj_ids.remove(0)
            if 255 in possible_obj_ids: possible_obj_ids.remove(255)
            obj_ids = possible_obj_ids
            if len(obj_ids) == 0:
                obj_ids = [0]
        
        for obj_id in obj_ids:
            obj_msk = msk.clone()
            # setting background 
            obj_msk[obj_msk != obj_id] = 0 
            # setting forground
            obj_msk[obj_msk == obj_id] = 1
            if obj_id == 0:
                obj_msk.zero_()
            # covert to float tensor to be processed by model
            masks_dict[obj_id] = obj_msk.type(torch.cuda.LongTensor)
        return masks_dict, obj_ids
    
    ### Inference through deep model
    def inference(self, model, seq, masks_id):
        initialized = False
        t = 1
        palette = None
        out_size = (480, 854)
        mask_list = []
        previous_msks = None
        if self.store_results:
            if not os.path.exists('{}/{}/{}'.format(self.res_root, 'overlay', seq)):
                os.mkdir('{}/{}/{}'.format(self.res_root, 'overlay', seq))
            if not os.path.exists('{}/{}/{}'.format(self.res_root, 'results', seq)):
                os.mkdir('{}/{}/{}'.format(self.res_root, 'results', seq))

        for img, msk in tqdm(list(self.dataset.get_frames(seq)), desc=seq, leave=False):
            img_tensor = torch.from_numpy(img.astype(np.float32) * (1.0 / 255.0)).permute(2,0,1).unsqueeze(0).cuda()
            #
            _, _, h, w = img_tensor.size()
            ph = int(math.floor(math.ceil(h / 16.0) * 16.0))
            pw = int(math.floor(math.ceil(w / 16.0) * 16.0))
            img_tensor = torch.nn.functional.interpolate(input=img_tensor, size=(ph, pw), mode='bilinear', align_corners=False)

            if not initialized:
                palette = msk.getpalette() or self._get_palette()
                msk_tensor = self._LabelToLongTensor(msk).unsqueeze(0).cuda()
                #print(msk_tensor.size())
                msk_dict, obj_ids = self._msk2dict(msk_tensor)
                with torch.no_grad():
                    #start_time = time.time()
                    _ = model.initialization(img_tensor, msk_dict)
                    #foreward_time = time.time()-start_time
                previous_msks = msk_dict
                mask_list.append(np.array(msk))
                initialized = True
            else:
                with torch.no_grad():
                    #start_time = time.time()
                    _, _, aggregated_seg, aggregated_seg_dict = model._inference(img_tensor, previous_msks, time_step=t, out_size = (h,w)) 
                    #foreward_time = time.time()-start_time
                pred_msk = aggregated_seg[0, 0,...].cpu().numpy()
                mask_list.append(pred_msk)
                previous_msks = aggregated_seg_dict
            
            #print('Using time %04f'%(foreward_time))
            
            if self.store_results:
                tar_mask = mask_list[-1]#[:,:,np.newaxis]
                self._save_mask(tar_mask, os.path.join(self.res_root, 'results', seq, '%05d.png'%(t-1)), colors=palette)
                self._overlay_davis(img, tar_mask, os.path.join(self.res_root, 'overlay', seq, '%05d.jpg'%(t-1)),colors=palette, alpha=0.7)
                
            t += 1
        #print(len(mask_list))
        masks = np.zeros((len(masks_id), *mask_list[0].shape))
        for i, m in enumerate(mask_list[1:-1]):
            masks[i, ...] = m
        num_objects = int(np.max(masks))
        tmp = np.ones((num_objects, *masks.shape))
        tmp = tmp * np.arange(1, num_objects + 1)[:, None, None, None]
        masks = (tmp == masks[None, ...]) > 0
        return masks

    def evaluate(self, model=None, res_path=None, metric=('J', 'F'), debug=False):
        if model is None and res_path is None:
            raise ValueError('Model and Results Path are both NoneTye')
        
        metric = metric if isinstance(metric, tuple) or isinstance(metric, list) else [metric]
        if 'T' in metric:
            raise ValueError('Temporal metric not supported!')
        if 'J' not in metric and 'F' not in metric:
            raise ValueError('Metric possible values are J for IoU or F for Boundary')

        # Containers
        metrics_res = {}
        if 'J' in metric:
            metrics_res['J'] = {"M": [], "R": [], "D": [], "M_per_object": {}, 'per_obj_frame': {}}
        if 'F' in metric:
            metrics_res['F'] = {"M": [], "R": [], "D": [], "M_per_object": {}, 'per_obj_frame': {}}

        # Sweep all sequences       
        results = Results(root_dir=res_path)

        for seq in tqdm(list(self.dataset.get_sequences())):
            all_gt_masks, all_void_masks, all_masks_id = self.dataset.get_all_masks(seq, True)
            #print(len(all_masks_id))
            if self.task == 'semi-supervised':
                all_gt_masks, all_masks_id = all_gt_masks[:, 1:-1, :, :], all_masks_id[1:-1]
            
            if res_path is not None:
                all_res_masks = results.read_masks(seq, all_masks_id)
            else:
                print('Get all results with trained model for seq [{}]'.format(seq))
                all_res_masks = self.inference(model, seq, all_masks_id)
            
            if self.task == 'unsupervised':
                j_metrics_res, f_metrics_res = self._evaluate_unsupervised(all_gt_masks, all_res_masks, all_void_masks, metric)
            elif self.task == 'semi-supervised':
                j_metrics_res, f_metrics_res = self._evaluate_semisupervised(all_gt_masks, all_res_masks, None, metric)
            for ii in range(all_gt_masks.shape[0]):
                seq_name = f'{seq}_{ii+1}'
                if 'J' in metric:
                    [JM, JR, JD] = utils.db_statistics(j_metrics_res[ii])
                    metrics_res['J']["M"].append(JM)
                    metrics_res['J']["R"].append(JR)
                    metrics_res['J']["D"].append(JD)
                    metrics_res['J']["M_per_object"][seq_name] = JM
                    metrics_res['J']["per_obj_frame"][seq_name] = j_metrics_res[ii]
                if 'F' in metric:
                    [FM, FR, FD] = utils.db_statistics(f_metrics_res[ii])
                    metrics_res['F']["M"].append(FM)
                    metrics_res['F']["R"].append(FR)
                    metrics_res['F']["D"].append(FD)
                    metrics_res['F']["M_per_object"][seq_name] = FM
                    metrics_res['F']["per_obj_frame"][seq_name] = f_metrics_res[ii]

            # Show progress
            if debug:
                sys.stdout.write(seq + '\n')
                sys.stdout.flush()
        return metrics_res
