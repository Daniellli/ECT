



from PIL import Image
import time
import os
import os.path as osp
import glob
from loguru import logger
import torch 
import cv2
from os.path import split,join
import torch
import numpy as np
import skimage
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import scipy.io as scio
from PIL import Image
import json
import shutil
from skimage import morphology


''' 
description:  将图像划分成16个部分, 
param {*} img
param {*} target_path
param {*} color
return {*}
'''
def split_to_16_part(img,save_path,color=[255,255,255]):
    H,W,C=img.shape
    tmp = img.copy()

    part_h = H//4
    part_w = W//4
    
    #* 绘制横白线
    for i in range(1,4):
        tmp[part_h*i:(part_h*i+5),:] =  color
    for i in range(1,4):
        tmp[:,part_w*i:(part_w*i+5)] =  color
    
    cv2.imwrite(save_path,tmp)
    
        

''' 
description:  对goal 的非零元素进行扩张, 扩张倍数为10 
param {*} goal 
param {*} times
return {*}
'''
def dilation(goal, times = 2 ):
    selem = skimage.morphology.disk(times)
    # goal = skimage.morphology.binary_dilation(goal, selem) != True
    goal = morphology.binary_dilation(goal, selem) != True
    goal = 1 - goal * 1.
    goal*=255
    return goal





'''
description: 
param {*} ims : 要拼接的两张图像
return {*}
'''
def pinjie(ims,save_name):

    # 单幅图像尺寸
    width, height = ims[0].size
 
    # 创建空白长图
    result = Image.new(ims[0].mode, (width, height * len(ims)))
 
    # 拼接图片
    for i, im in enumerate(ims):
        result.paste(im, box=(0, i * height))
 
    # 保存图片
    result.save(save_name)



def load_json(path):
    with open(path,'r') as f:
        data = json.load(f)
    return data




'''
description:  读取mat格式的gt
param {*} path
return {*}
'''
def read_mat_gt(path):
    if  not osp.exists(path):
        print(f"path no exists")
    gt_mask = scio.loadmat(path)
    gt_mask = gt_mask['groundTruth'][0][0][0][0][0]
    return gt_mask
    
    