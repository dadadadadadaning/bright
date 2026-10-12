import argparse
import os
import random

import imageio
import numpy as np
from torch.utils.data import Dataset
## import imutils


def img_loader(path):
    img = np.array(imageio.imread(path), np.float32)
    return img


def normalize_img(img):
    """
    ImageNet normalization.
    Input: H,W,3, value range usually 0-255.
    Output: H,W,3 float32.
    """
    img = img.astype(np.float32)

    if img.max() > 1.0:
        img = img / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    img = (img - mean) / std
    return img.astype(np.float32)


class MultimodalDamageAssessmentDatset(Dataset):
    """
    ROI Mask 版本数据加载器。

    关键修改：
    1. 不再从 dataset/building-mask/ 读取额外 mask 文件；
    2. 直接从 target 标签生成建筑 mask：
       bmask = (clf_label > 0)
    3. bmask 与 pre_img、post_img、clf_label 一起参与随机裁剪、翻转、旋转；
    4. 返回顺序保持为：
       pre_img, post_img, bmask, loc_label, clf_label, data_idx
    """

    def __init__(
        self,
        dataset_path,
        data_list,
        crop_size,
        max_iters=None,
        type='train',
        data_loader=img_loader,
        suffix='.tif'
    ):
        self.dataset_path = dataset_path
        self.data_list = data_list
        self.loader = data_loader
        self.type = type
        self.data_pro_type = self.type
        self.suffix = suffix

        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))
            self.data_list = self.data_list[0:max_iters]

        self.crop_size = crop_size

    def __random_crop(self, pre_img, post_img, label, bmask, crop_size, mean_rgb=[0, 0, 0], ignore_index=255):
        h, w = label.shape

        H = max(crop_size, h)
        W = max(crop_size, w)

        pad_pre_image = np.zeros((H, W, 3), dtype=np.float32)
        pad_post_image = np.zeros((H, W, 3), dtype=np.float32)
        pad_label = np.ones((H, W), dtype=np.float32) * ignore_index
        pad_bmask = np.zeros((H, W), dtype=np.float32)

        pad_pre_image[:, :, 0] = mean_rgb[0]
        pad_pre_image[:, :, 1] = mean_rgb[1]
        pad_pre_image[:, :, 2] = mean_rgb[2]

        pad_post_image[:, :, 0] = mean_rgb[0]
        pad_post_image[:, :, 1] = mean_rgb[1]
        pad_post_image[:, :, 2] = mean_rgb[2]

        H_pad = int(np.random.randint(H - h + 1))
        W_pad = int(np.random.randint(W - w + 1))

        pad_pre_image[H_pad:(H_pad + h), W_pad:(W_pad + w), :] = pre_img
        pad_post_image[H_pad:(H_pad + h), W_pad:(W_pad + w), :] = post_img
        pad_label[H_pad:(H_pad + h), W_pad:(W_pad + w)] = label
        pad_bmask[H_pad:(H_pad + h), W_pad:(W_pad + w)] = bmask

        def get_random_cropbox(cat_max_ratio=0.75):
            # 默认兜底，避免极端情况下循环未进入导致变量未定义
            H_start = random.randrange(0, H - crop_size + 1, 1)
            W_start = random.randrange(0, W - crop_size + 1, 1)
            H_end = H_start + crop_size
            W_end = W_start + crop_size

            for _ in range(10):
                H_start = random.randrange(0, H - crop_size + 1, 1)
                H_end = H_start + crop_size
                W_start = random.randrange(0, W - crop_size + 1, 1)
                W_end = W_start + crop_size

                temp_label = pad_label[H_start:H_end, W_start:W_end]
                index, cnt = np.unique(temp_label, return_counts=True)

                # 去掉 ignore_index
                cnt = cnt[index != ignore_index]

                # 原代码 len(cnt > 1) 写法不严谨，这里改为 len(cnt) > 1
                if len(cnt) > 1 and np.max(cnt) / np.sum(cnt) < cat_max_ratio:
                    break

            return H_start, H_end, W_start, W_end

        H_start, H_end, W_start, W_end = get_random_cropbox()

        pre_img = pad_pre_image[H_start:H_end, W_start:W_end, :]
        post_img = pad_post_image[H_start:H_end, W_start:W_end, :]
        label = pad_label[H_start:H_end, W_start:W_end]
        bmask = pad_bmask[H_start:H_end, W_start:W_end]

        return pre_img, post_img, label, bmask

    def __transforms(self, aug, pre_img, post_img, label, bmask):
        if aug:
            pre_img, post_img, label, bmask = self.__random_crop(
                pre_img,
                post_img,
                label,
                bmask,
                self.crop_size
            )

            if random.random() > 0.5:
                pre_img = np.fliplr(pre_img).copy()
                post_img = np.fliplr(post_img).copy()
                label = np.fliplr(label).copy()
                bmask = np.fliplr(bmask).copy()

            if random.random() > 0.5:
                pre_img = np.flipud(pre_img).copy()
                post_img = np.flipud(post_img).copy()
                label = np.flipud(label).copy()
                bmask = np.flipud(bmask).copy()

            k = random.randrange(3) + 1
            pre_img = np.rot90(pre_img, k).copy()
            post_img = np.rot90(post_img, k).copy()
            label = np.rot90(label, k).copy()
            bmask = np.rot90(bmask, k).copy()

        pre_img = normalize_img(pre_img)
        pre_img = np.transpose(pre_img, (2, 0, 1)).astype(np.float32)

        post_img = normalize_img(post_img)
        post_img = np.transpose(post_img, (2, 0, 1)).astype(np.float32)

        # bmask: [H,W] -> [1,H,W]
        bmask = (np.asarray(bmask) > 0).astype(np.float32)
        bmask = np.expand_dims(bmask, axis=0)

        # label 保持 [H,W]
        label = np.asarray(label).astype(np.float32)

        return pre_img, post_img, bmask, label

    def __getitem__(self, index):
        data_idx = self.data_list[index]

        pre_path = os.path.join(
            self.dataset_path,
            'pre-event',
            data_idx + '_pre_disaster' + self.suffix
        )
        post_path = os.path.join(
            self.dataset_path,
            'post-event',
            data_idx + '_post_disaster' + self.suffix
        )
        label_path = os.path.join(
            self.dataset_path,
            'target',
            data_idx + '_building_damage' + self.suffix
        )

        pre_img = self.loader(pre_path)[:, :, 0:3]
        post_img = self.loader(post_path)
        clf_label = self.loader(label_path)

        # 防止 label 是三通道图
        if clf_label.ndim == 3:
            clf_label = clf_label[:, :, 0]

        # SAR 单通道复制成三通道，适配 ImageNet/Swin 输入
        if post_img.ndim == 2:
            post_img = np.stack((post_img,) * 3, axis=-1)
        elif post_img.ndim == 3:
            post_img = post_img[:, :, 0]
            post_img = np.stack((post_img,) * 3, axis=-1)

        # 核心修改：不再读取 building-mask 文件夹
        # 直接由 target 生成建筑 mask。0=背景，1/2/3=建筑。
        bmask = (clf_label > 0).astype(np.float32)

        if 'train' in self.data_pro_type:
            pre_img, post_img, bmask, clf_label = self.__transforms(
                True,
                pre_img,
                post_img,
                clf_label,
                bmask
            )
        else:
            pre_img, post_img, bmask, clf_label = self.__transforms(
                False,
                pre_img,
                post_img,
                clf_label,
                bmask
            )
            clf_label = np.asarray(clf_label)

        # 定位标签：0=背景，1=建筑
        loc_label = clf_label.copy()
        loc_label[loc_label == 2] = 1
        loc_label[loc_label == 3] = 1
        loc_label[loc_label == 255] = 255

        return pre_img, post_img, bmask, loc_label, clf_label, data_idx

    def __len__(self):
        return len(self.data_list)


class MultimodalDamageAssessmentDatset_Inference(Dataset):
    def __init__(self, dataset_path, data_list, data_loader=img_loader, suffix='.tif'):
        self.dataset_path = dataset_path
        self.data_list = data_list
        self.loader = data_loader
        self.suffix = suffix

    def __transforms(self, pre_img, post_img):
        pre_img = normalize_img(pre_img)
        pre_img = np.transpose(pre_img, (2, 0, 1)).astype(np.float32)

        post_img = normalize_img(post_img)
        post_img = np.transpose(post_img, (2, 0, 1)).astype(np.float32)

        return pre_img, post_img

    def __getitem__(self, index):
        data_idx = self.data_list[index]

        pre_path = os.path.join(
            self.dataset_path,
            'pre-event',
            data_idx + '_pre_disaster' + self.suffix
        )
        post_path = os.path.join(
            self.dataset_path,
            'post-event',
            data_idx + '_post_disaster' + self.suffix
        )

        pre_img = self.loader(pre_path)[:, :, 0:3]
        post_img = self.loader(post_path)

        if post_img.ndim == 2:
            post_img = np.stack((post_img,) * 3, axis=-1)
        elif post_img.ndim == 3:
            post_img = post_img[:, :, 0]
            post_img = np.stack((post_img,) * 3, axis=-1)

        pre_img, post_img = self.__transforms(pre_img, post_img)

        return pre_img, post_img, data_idx

    def __len__(self):
        return len(self.data_list)
