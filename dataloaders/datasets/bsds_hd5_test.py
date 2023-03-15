import os
import torch
import h5py
import random
import numpy as np
import scipy.io
from tqdm import tqdm
from torch.utils.data import DataLoader
from PIL import Image,ImageStat
from torch.utils import data
from torchvision import transforms



class MydatasetTest(data.Dataset):

    def __init__(self,root_path='....../Augmentation/', split='trainval',crop_size=513):
        self.split = split
        self.crop_size=crop_size
        list_file = os.path.join(root_path, 'test.lst')

        with open(list_file, 'r') as f:
            lines = f.readlines()
        lines = [line.strip() for line in lines]


        self.images_path = lines
        self.images_name = []
        for path in self.images_path:
            folder, filename = os.path.split(path)
            name, ext = os.path.splitext(filename)
            self.images_name.append(name)

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.trans = transforms.Compose([
                             transforms.ToTensor(),
                             normalize])  
        


    def __len__(self):
        return len(self.images_path)

    '''
    description:  1.读取图片, 2.计算图像中心 3. 读取标签,  4.将图像和标签裁剪成方格形的数据格式 5.数据增强, 6.转tensor 7.返回
    #? 这个标签如何生成的 [5,w,h]  ,这个5个索引分别对应什么? 
    param {*} self
    param {*} idx
    return {*}
    '''
    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.images_path[idx])).convert('RGB')
        img_tensor = self.trans(img)
        return img_tensor

    




if __name__ == '__main__':
    train_dataset = Mydataset(root_path='/*******/Augmentation/', split='trainval', crop_size=320)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True,
                                   num_workers=4, pin_memory=True, drop_last=True)
    tbar = tqdm(train_loader)
    num_img_tr = len(train_loader)
    for i, (image, target) in enumerate(tbar):
        print(target.shape)







