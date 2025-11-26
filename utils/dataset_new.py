import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
import torchvision.transforms as Transforms
import numpy as np
import random

class ISBI_Loader(Dataset):
    def __init__(self, data_path, transform=None):
        # load data_path
        self.data_path = data_path
        # self.imgs_path = glob.glob(os.path.join(data_path, 'image/*.png'))

        # 分别匹配 .png 和 .tif 文件
        png_imgs = glob.glob(os.path.join(data_path, 'image/*.png'))
        tif_imgs = glob.glob(os.path.join(data_path, 'image/*.tif'))
        # 合并两个列表
        self.imgs_path = png_imgs + tif_imgs

        png_lbs = glob.glob(os.path.join(data_path, 'label/*.png'))
        tif_lbs = glob.glob(os.path.join(data_path, 'label/*.tif'))
        # 合并两个列表
        self.labels_path = png_lbs + tif_lbs

        self.transform = transform

    def augment(self, image, flipCode):
        # using cv2.flip to aug image
        flip = cv2.flip(image, flipCode)
        return flip

    def __getitem__(self, intex):
        image1_path = self.imgs_path[intex]
        label_path = self.labels_path[intex]
        # label_path = image1_path.replace('image', 'label')

        image1 = cv2.imread(image1_path)
        label = cv2.imread(label_path)

        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        # label[label == 255] = 1
        label = label.reshape(label.shape[0], label.shape[1], 1)

        fimage = self.transform(image1)
        flabel = self.transform(label)

        return fimage, flabel

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)

def mixup_collate_fn(alpha):
    def _collate(batch):
        imgs, labs = list(zip(*batch))          # list of tensor
        imgs = torch.stack(imgs, dim=0)         # B,C,H,W
        labs = torch.stack(labs, dim=0)         # B,1,H,W 或 B,H,W
        batch_size = imgs.size(0)
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.
        lam = max(lam, 1. - lam)

        indices = torch.randperm(batch_size)
        imgs_mixed = lam * imgs + (1 - lam) * imgs[indices]
        labs_mixed = lam * labs + (1 - lam) * labs[indices]
        return imgs_mixed, labs_mixed
    return _collate
