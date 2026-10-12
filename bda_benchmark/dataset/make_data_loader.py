import argparse
import os

import imageio
import numpy as np
from torch.utils.data import Dataset
# import cv2 
from bda_benchmark.dataset import imutils
import matplotlib.pyplot as plt
from torch.utils import data
from PIL import Image


def img_loader(path):
    img = np.array(imageio.imread(path), np.float32)
    return img



class MultimodalDamageAssessmentDatset(Dataset):
    def __init__(
        self,
        dataset_path,
        data_list,
        crop_size,
        max_iters=None,
        type='train',
        data_loader=img_loader,
        suffix='.tif',
        use_damage_aware_sampler=False,
        damage_crop_prob=0.5,
        intact_crop_prob=0.25,
        random_crop_prob=0.25,
        min_damage_pixels=20,
        max_crop_retry=20,
    ):
        self.dataset_path = dataset_path
        self.data_list = data_list
        self.loader = data_loader
        self.type = type
        self.data_pro_type = self.type
        self.suffix = suffix
        self.use_damage_aware_sampler = bool(use_damage_aware_sampler)
        self.damage_crop_prob = float(damage_crop_prob)
        self.intact_crop_prob = float(intact_crop_prob)
        self.random_crop_prob = float(random_crop_prob)
        self.min_damage_pixels = int(min_damage_pixels)
        self.max_crop_retry = int(max_crop_retry)

        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))
            self.data_list = self.data_list[0:max_iters]
        self.crop_size = crop_size

    def __transforms(self, aug, pre_img, post_img, label):
        if aug:
            if self.use_damage_aware_sampler:
                pre_img, post_img, label = imutils.damage_aware_random_crop(
                    pre_img,
                    post_img,
                    label,
                    self.crop_size,
                    damage_crop_prob=self.damage_crop_prob,
                    intact_crop_prob=self.intact_crop_prob,
                    random_crop_prob=self.random_crop_prob,
                    min_damage_pixels=self.min_damage_pixels,
                    max_crop_retry=self.max_crop_retry,
                )
            else:
                pre_img, post_img, label = imutils.random_crop(pre_img, post_img, label, self.crop_size)
            pre_img, post_img, label = imutils.random_fliplr(pre_img, post_img, label)
            pre_img, post_img, label = imutils.random_flipud(pre_img, post_img, label)
            pre_img, post_img, label = imutils.random_rot(pre_img, post_img, label)

        pre_img = imutils.normalize_img(pre_img)  # imagenet normalization
        pre_img = np.transpose(pre_img, (2, 0, 1))

        post_img = imutils.normalize_img(post_img)  # imagenet normalization
        post_img = np.transpose(post_img, (2, 0, 1))

        return pre_img, post_img, label

    def __getitem__(self, index):
        pre_path = os.path.join(self.dataset_path, 'pre-event', self.data_list[index] + '_pre_disaster' + self.suffix)
        post_path = os.path.join(self.dataset_path, 'post-event', self.data_list[index] + '_post_disaster'  + self.suffix)
        label_path = os.path.join(self.dataset_path, 'target', self.data_list[index] + '_building_damage'  + self.suffix)
        pre_img = self.loader(pre_path)[:,:,0:3] 
        post_img = self.loader(post_path)  
        
        # pre_img = np.stack((pre_img,)*3, axis=-1)
        post_img = np.stack((post_img,)*3, axis=-1)
        clf_label = self.loader(label_path)
        

        if 'train' in self.data_pro_type:
            pre_img, post_img, clf_label = self.__transforms(True, pre_img, post_img, clf_label)
        else:
            pre_img, post_img, clf_label = self.__transforms(False, pre_img, post_img, clf_label)
            clf_label = np.asarray(clf_label)
        loc_label = clf_label.copy()
        loc_label[loc_label == 2] = 1
        loc_label[loc_label == 3] = 1

        data_idx = self.data_list[index]
        return pre_img, post_img, loc_label, clf_label, data_idx

    def __len__(self):
        return len(self.data_list)


class MultimodalDamageAssessmentDatset_Inference(Dataset):
    def __init__(self, dataset_path, data_list, data_loader=img_loader, suffix='.tif'):
        self.dataset_path = dataset_path
        self.data_list = data_list
        self.loader = data_loader
        self.suffix = suffix

    def __transforms(self, pre_img, post_img):
        pre_img = imutils.normalize_img(pre_img)  # imagenet normalization
        pre_img = np.transpose(pre_img, (2, 0, 1))

        post_img = imutils.normalize_img(post_img)  # imagenet normalization
        post_img = np.transpose(post_img, (2, 0, 1))

        return pre_img, post_img

    def __getitem__(self, index):
        pre_path = os.path.join(self.dataset_path, 'pre-event', self.data_list[index] + '_pre_disaster' + self.suffix)
        post_path = os.path.join(self.dataset_path, 'post-event', self.data_list[index] + '_post_disaster'  + self.suffix)
        pre_img = self.loader(pre_path)[:,:,0:3] 
        post_img = self.loader(post_path)  
        
        # pre_img = np.stack((pre_img,)*3, axis=-1)
        post_img = np.stack((post_img,)*3, axis=-1) 
        
        pre_img, post_img = self.__transforms(pre_img, post_img)
    
        data_idx = self.data_list[index]
        return pre_img, post_img, data_idx
    
    def __len__(self):
        return len(self.data_list)
