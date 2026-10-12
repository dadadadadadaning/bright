import os
import numpy as np
import torch
from skimage import io
from torch.utils import data
import utils.transform as transform
import matplotlib.pyplot as plt
from skimage.transform import rescale
from torchvision.transforms import functional as F
import random
from PIL import Image
from PIL import ImageFilter
from torchvision import transforms
# from osgeo import gdal_array
import cv2
from skimage import io, img_as_float
from scipy.ndimage import uniform_filter


MEAN_A = np.array([113.40, 114.08, 116.45])
STD_A  = np.array([48.30,  46.27,  48.14])
MEAN_B = np.array([111.07, 111.07, 111.07])
STD_B  = np.array([49.41,  49.41,  49.41])

root = '/data/jingwei/HeteCD/data/xiongan_data_old'

class CDDataAugmentation:

    def __init__(
            self,
            img_size,
            with_random_hflip=False,
            with_random_vflip=False,
            with_random_rot=False,
            with_random_crop=False,
            with_scale_random_crop=False,
            with_random_blur=False,
            random_color_tf=False
    ):
        self.img_size = img_size
        if self.img_size is None:
            self.img_size_dynamic = True
        else:
            self.img_size_dynamic = False
        self.with_random_hflip = with_random_hflip
        self.with_random_vflip = with_random_vflip
        self.with_random_rot = with_random_rot
        self.with_random_crop = with_random_crop
        self.with_scale_random_crop = with_scale_random_crop
        self.with_random_blur = with_random_blur
        self.random_color_tf=random_color_tf
    def transform(self, imgs, labels, to_tensor=True):
        """
        :param imgs: [ndarray,]
        :param labels: [ndarray,]
        :return: [ndarray,],[ndarray,]
        """
        # resize image and covert to tensor
        imgs = [F.to_pil_image(img) for img in imgs]
        if self.img_size is None:
            self.img_size = None

        if not self.img_size_dynamic:
            if imgs[0].size != (self.img_size, self.img_size):
                imgs = [F.resize(img, [self.img_size, self.img_size], interpolation=3)
                        for img in imgs]
        else:
            self.img_size = imgs[0].size[0]

        labels = [F.to_pil_image(img) for img in labels]
        if len(labels) != 0:
            if labels[0].size != (self.img_size, self.img_size):
                labels = [F.resize(img, [self.img_size, self.img_size], interpolation=0)
                        for img in labels]

        random_base = 0.5
        if self.with_random_hflip and random.random() > 0.5:
            imgs = [F.hflip(img) for img in imgs]
            labels = [F.hflip(img) for img in labels]

        if self.with_random_vflip and random.random() > 0.5:
            imgs = [F.vflip(img) for img in imgs]
            labels = [F.vflip(img) for img in labels]

        if self.with_random_rot and random.random() > random_base:
            angles = [90, 180, 270]
            index = random.randint(0, 2)
            angle = angles[index]
            imgs = [F.rotate(img, angle) for img in imgs]
            labels = [F.rotate(img, angle) for img in labels]

        if self.with_random_crop and random.random() > 0:
            i, j, h, w = transforms.RandomResizedCrop(size=self.img_size). \
                get_params(img=imgs[0], scale=(0.8, 1.2), ratio=(1, 1))

            imgs = [F.resized_crop(img, i, j, h, w,
                                    size=(self.img_size, self.img_size),
                                    interpolation=Image.CUBIC)
                    for img in imgs]

            labels = [F.resized_crop(img, i, j, h, w,
                                      size=(self.img_size, self.img_size),
                                      interpolation=Image.NEAREST)
                      for img in labels]

        if self.with_scale_random_crop:
            # rescale
            scale_range = [1, 1.2]
            target_scale = scale_range[0] + random.random() * (scale_range[1] - scale_range[0])

            imgs = [pil_rescale(img, target_scale, order=3) for img in imgs]
            labels = [pil_rescale(img, target_scale, order=0) for img in labels]
            # crop
            imgsize = imgs[0].size  # h, w
            box = get_random_crop_box(imgsize=imgsize, cropsize=self.img_size)
            imgs = [pil_crop(img, box, cropsize=self.img_size, default_value=0)
                    for img in imgs]
            labels = [pil_crop(img, box, cropsize=self.img_size, default_value=255)
                    for img in labels]

        if self.with_random_blur and random.random() > 0:
            radius = random.random()
            imgs = [img.filter(ImageFilter.GaussianBlur(radius=radius))
                    for img in imgs]

        if self.random_color_tf:
            color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3)
            imgs_tf = []
            for img in imgs:
                tf = transforms.ColorJitter(
                            color_jitter.brightness, 
                            color_jitter.contrast, 
                            color_jitter.saturation,
                            color_jitter.hue)
                imgs_tf.append(tf(img))
            imgs = imgs_tf
            
        # if to_tensor:
        #     # to tensor
        #     imgs = [TF.to_tensor(img) for img in imgs]
        labels = [torch.from_numpy(np.array(img, np.uint8)).unsqueeze(dim=0)
                for img in labels]
            
        #     imgs = [TF.normalize(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        #             for img in imgs]
        # img1 = np.array(imgs[0])
        imgs = [F.to_tensor(img) for img in imgs]
        return imgs, labels


def pil_crop(image, box, cropsize, default_value):
    assert isinstance(image, Image.Image)
    img = np.array(image)

    if len(img.shape) == 3:
        cont = np.ones((cropsize, cropsize, img.shape[2]), img.dtype)*default_value
    else:
        cont = np.ones((cropsize, cropsize), img.dtype)*default_value
    cont[box[0]:box[1], box[2]:box[3]] = img[box[4]:box[5], box[6]:box[7]]

    return Image.fromarray(cont)


def get_random_crop_box(imgsize, cropsize):
    h, w = imgsize
    ch = min(cropsize, h)
    cw = min(cropsize, w)

    w_space = w - cropsize
    h_space = h - cropsize

    if w_space > 0:
        cont_left = 0
        img_left = random.randrange(w_space + 1)
    else:
        cont_left = random.randrange(-w_space + 1)
        img_left = 0

    if h_space > 0:
        cont_top = 0
        img_top = random.randrange(h_space + 1)
    else:
        cont_top = random.randrange(-h_space + 1)
        img_top = 0

    return cont_top, cont_top+ch, cont_left, cont_left+cw, img_top, img_top+ch, img_left, img_left+cw


def pil_rescale(img, scale, order):
    assert isinstance(img, Image.Image)
    height, width = img.size
    target_size = (int(np.round(height*scale)), int(np.round(width*scale)))
    return pil_resize(img, target_size, order)


def pil_resize(img, size, order):
    assert isinstance(img, Image.Image)
    if size[0] == img.size[0] and size[1] == img.size[1]:
        return img
    if order == 3:
        resample = Image.BICUBIC
    elif order == 0:
        resample = Image.NEAREST
    return img.resize(size[::-1], resample)

def showIMG(img):
    plt.imshow(img)
    plt.show()
    return 0
def lee_filter(img, size):
    img = img_as_float(img)  # 转换图像到浮点型，范围在0到1
    mean_img = uniform_filter(img, size)  # 计算局部均值
    mean_sqr_img = uniform_filter(img**2, size)  # 计算局部平方均值
    var_img = mean_sqr_img - mean_img**2  # 计算局部方差
    noise = np.mean(var_img)
    coeff = var_img / (var_img + noise)
    result_img = mean_img + coeff * (img - mean_img)
    return result_img
def normalize_image(im, time='A'):
    assert time in ['A', 'B']
    im = im/255
    return im

def read_RSimages(mode, rescale=False):
    #assert mode in ['train', 'val', 'train_unchange']
    img_A_dir = os.path.join(root, mode, 'A')
    img_B_dir = os.path.join(root, mode, 'B')
    label_A_dir = os.path.join(root, mode, 'label')
    # To use rgb labels:
    #label_A_dir = os.path.join(root, mode, 'label1_rgb')
    #label_B_dir = os.path.join(root, mode, 'label2_rgb')
    
    data_list = os.listdir(img_A_dir)
    imgs_list_A, imgs_list_B, labels_A, labels_B = [], [], [], []
    count = 0
    for it in data_list:
        # print(it)
        # if (it[-4:]=='.tif'):
        img_A_path = os.path.join(img_A_dir, it)
        img_B_path = os.path.join(img_B_dir, it)
        label_A_path = os.path.join(label_A_dir, it)
        imgs_list_A.append(img_A_path)
        imgs_list_B.append(img_B_path)
        label_A = io.imread(label_A_path)
        labels_A.append(label_A)
        count+=1
        if not count%500: print('%d/%d images loaded.'%(count, len(data_list)))
    
    print(labels_A[0].shape)
    print(str(len(imgs_list_A)) + ' ' + mode + ' images' + ' loaded.')
    
    return imgs_list_A, imgs_list_B, labels_A

class Data(data.Dataset):
    def __init__(self, mode, random_flip = False):
        self.random_flip = random_flip
        self.imgs_list_A, self.imgs_list_B, self.labels = read_RSimages(mode)
        self.mode = mode
        self.augm = CDDataAugmentation(
                    img_size=512,
                    with_random_hflip=True,
                    with_random_vflip=True,
                    with_scale_random_crop=True,
                    with_random_blur=True,
                    random_color_tf=True
                )
    def get_mask_name(self, idx):
        mask_name = os.path.split(self.imgs_list_A[idx])[-1]
        return mask_name

    def __getitem__(self, idx):
        img_A = io.imread(self.imgs_list_A[idx])
        name = self.imgs_list_A[idx].split('/')[-1].replace('.tif','.png')
        img_B = io.imread(self.imgs_list_B[idx])
        label= self.labels[idx]//255
        if self.mode=="train":
            [img_A, img_B], [label] = self.augm.transform([img_A, img_B], [label])
        else:
            img_A = F.to_tensor(img_A)
            img_B = F.to_tensor(img_B)
            label = torch.from_numpy(np.array(label, np.uint8)).unsqueeze(dim=0)
        
        # img_B = img_B/255
        # img_A = img_A/255  

        return img_A, img_B, label.squeeze(), name

    def __len__(self):
        return len(self.imgs_list_A)

class Data_test(data.Dataset):
    def __init__(self, test_dir):
        self.imgs_A = []
        self.imgs_B = []
        self.mask_name_list = []
        imgA_dir = os.path.join(test_dir, 'pre')
        imgB_dir = os.path.join(test_dir, 'post')
        data_list = os.listdir(imgA_dir)
        for it in data_list:
            if (it[-4:]=='.png'):
                img_A_path = os.path.join(imgA_dir, it)
                img_B_path = os.path.join(imgB_dir, it)
                self.imgs_A.append(io.imread(img_A_path))
                self.imgs_B.append(io.imread(img_B_path))
                self.mask_name_list.append(it)
        self.len = len(self.imgs_A)

    def get_mask_name(self, idx):
        return self.mask_name_list[idx]

    def __getitem__(self, idx):
        img_A = self.imgs_A[idx]
        img_B = self.imgs_B[idx]
        img_A = normalize_image(img_A, 'A')
        img_B = normalize_image(img_B, 'B')
        return F.to_tensor(img_A), F.to_tensor(img_B)

    def __len__(self):
        return self.len

