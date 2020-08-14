import os
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import random
import cv2
from PIL import Image

import time
import mat4py
import numpy as np
import pandas as pd
color_to_label={'unlabeled':0,'Black_Footed_Albatross':250,'Laysan_Albatross ':200,
                'Bank_Swallow ':150,'Crested_Auklet ':100,'Groove_Billed_Ani ':50}
full_to_train = {0: 0, 50: 1, 100:2, 150:3, 200:4, 250:5}
train_to_full = {0: 0, 1: 50, 2:100, 3:150, 4:200, 5:250}

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

img_transform = transforms.Compose([transforms.ToTensor(),])
BATCH_SIZE=12
num_classes=6
random.seed(1)

dataset_dir = "./dataset/data"

class MyDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        :param data_dir: str, 数据集所在路径
        :param transform: torch.transform，数据预处理
        """
        self.data_info = self.get_img_info(data_dir)  # data_info存储所有图片路径和标签，在DataLoader中通过index读取样本
        self.transform = transform

    def __getitem__(self, index):
        imgA, imgB, one_hot_mask, mask, category = self.data_info[index]
        if self.transform is not None:
            imgA = self.transform(imgA)   # 在这里做transform，转为tensor等等
        imgB = img_transform(imgB)
        return imgA, imgB, one_hot_mask, mask, category

    def __len__(self):
        return len(self.data_info)

    @staticmethod
    def get_img_info(data_dir):
        data_info = list()
        mat_path = os.path.join(data_dir, 'data')
        img_names = os.listdir(mat_path)
        for i in range(len(img_names)):
            img_name = img_names[i]
            category = img_name
            path_img = os.path.join(data_dir, 'data', img_name)
            path_mask = os.path.join(data_dir, 'mask', img_name[0:-4]+'.png')

            imgA = cv2.imread(path_img)
            imgA = cv2.resize(imgA, (256, 256))[16:240, 16: 240]
            imgB = imgA

            mask = cv2.imread(path_mask,0)
            mask = cv2.resize(mask, (256, 256),interpolation=cv2.INTER_NEAREST)[16:240, 16: 240]
            mask = mask.astype('uint8')
            mask_copy = mask.copy()
            w,h = mask_copy.shape

            for k, v in full_to_train.items():
                mask_copy[mask == k] = v
            mask_copy = mask_copy.astype('uint8')
            one_hot_mask = np.zeros((num_classes, w, h))

            for c in range(num_classes):  # 把taget变成 类别数x高x宽 ==>类别数x一个面
                one_hot_mask[c][mask_copy == c] = 1  # 每一类占一个面，原图里A类的像素点坐标(i,j)，那么在属于A类的(i,j)处设为1
            one_hot_mask = torch.FloatTensor(one_hot_mask)
            data_info.append((imgA, imgB, one_hot_mask, mask, category))

        return data_info


# 构建MyDataset实例
Dataset = MyDataset(data_dir='./dataset', transform=transform)
train_size = int(0.8 * len(Dataset))
test_size = len(Dataset) - train_size
train_dataset, test_dataset = random_split(Dataset, [train_size, test_size])

# 构建DataLoder
train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE)
