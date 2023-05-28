'''
Author:   "  "
Date: 2022-08-28 22:57:06
LastEditTime: 2022-09-06 13:42:08
LastEditors:   "  "
Description: 
FilePath: /Cerberus-main/plot-rind-edge-pr-curves/plot_attention.py
email:  
'''



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
from PIL import Image


import scipy.io as scio



def make_dir(path):
    if  not osp.exists(path):
        os.makedirs(path)

def plot_attention(all_images,gt_save_path,save_path):
    layer_num = 6

    for im in all_images:
        for t in TASKS[:-1]:
            concat_res = None
            for i in range(layer_num):
                
                if i == 0 :
                    a = cv2.imread(osp.join(edge_cerberus,"attention",im,f"atten-{t}-{i}.jpg"))
                    concat_res = np.concatenate([
                        cv2.imread(osp.join(gt_save_path,f"{im}_{t}.png"))[:a.shape[0],:,:],
                        np.ones([a.shape[0],10,3])*255,
                        a,
                        np.ones([a.shape[0],10,3])*255],1)
                else :
                    concat_res = np.concatenate([concat_res,
                                    cv2.imread(osp.join(edge_cerberus,"attention",im,f"atten-{t}-{i}.jpg")),
                                    np.ones([concat_res.shape[0],10,3])*255],1)

            
            before_nms = cv2.imread(osp.join(edge_cerberus,t,'png',f"{im}.png"))[:concat_res.shape[0],:]
            
            concat_res = np.concatenate([concat_res,
                                    before_nms,
                                    np.ones([concat_res.shape[0],10,3])*255,
                                    cv2.imread(osp.join(edge_cerberus,t,'nms',f"{im}.png"))[:concat_res.shape[0],:,:] ],1)
            cv2.imwrite(osp.join(save_path, f'{im}_{t}.png'),concat_res)


'''
description:  加载mat文件
param {*} path
return {*}
'''
def loadmat(path):
    return scio.loadmat(path)['result']

'''
description:  加载mat然后写入save_path
param {*} path
param {*} save_path
return {*}
'''
def mat2png_save(path,save_path):
    data  = loadmat(path)*255
    
    cv2.imwrite(save_path,data)

'''
description: 读取所有mat文件, 转png然后存储 
param {*} path
param {*} imgs
return {*}
'''
def mat2png(path,imgs):

    for task in TASKS:
        target_path = osp.join(path,task,'png')
        make_dir(target_path)
        for im in imgs:
            mat2png_save(osp.join(path,task,'met',f"{im}.mat"),
                        save_path= osp.join(target_path,f"{im}.png"))

    

    

SAVE_ROOT = "/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/plot-rind-edge-pr-curves/plot_material"
EVAL_RES_ROOT="/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/networks"
ORIGIN_IMG_PATH="/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/dataset/BSDS-RIND/test" #! 读取测试图像的路径
ORIGIN_IMG_GT = "/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/dataset/BSDS-RIND/testgt"
ORIGIN_IMG_GT2 = "/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/dataset/BSDS_RIND_mine"

#* 每种边缘对应的颜色
RINDE_COLOR = [
    (10,139,226),
    (142,217,199),
    (235,191,114),
    (174, 125, 176),
    (219, 118, 2)
]


COLORS = {
    "TP":(10,251,9),
    "FN":(8,9,250),
    # "FN":(255,255,0),#* 青色
    # "FN":(255,0,255),#* 深红色
    # "FN":(203,192,255),#* 粉色
    # "FN":(255,1,1),#* 蓝色
    "FP":(7,255,252),
}

TASKS = ["reflectance","illumination","normal","depth","all_edges"]

edge_cerberus= osp.join(EVAL_RES_ROOT,"final_version/edge_final_8_3090_0")
edge_cerberus_save_path =  osp.join(SAVE_ROOT,"Ours")

without_loss_path= osp.join(EVAL_RES_ROOT,"final_version/edge_final_4_A100_80G_no_loss_0")
without_loss_save_path =osp.join(SAVE_ROOT,"without_loss_path")

gt_save_path = osp.join(SAVE_ROOT,'GT')



#* concat attention 
all_images =sorted( [x.split('.')[0] for x in os.listdir(osp.join(ORIGIN_IMG_GT,'depth'))])




save_path = osp.join(SAVE_ROOT,'attention1')
save_path2 = osp.join(SAVE_ROOT,'attention_concat')
make_dir(save_path)
make_dir(save_path2)
gt_save_path = osp.join(SAVE_ROOT,'GT')
# mat2png(edge_cerberus,all_images)
plot_attention(all_images,gt_save_path,save_path)
for i in all_images:
    imgs = None
    for idx ,  task in enumerate(TASKS[:-1]):
        if idx == 0:
            imgs = cv2.imread(osp.join(save_path,f"{i}_{task}.png"))
            imgs = np.concatenate([imgs,np.ones([10,imgs.shape[1],3])*255])
        else :
            imgs = np.concatenate([imgs,
                            cv2.imread(osp.join(save_path,f"{i}_{task}.png")),
                            np.ones([10,imgs.shape[1],3])*255])
    cv2.imwrite(osp.join(save_path2, f'{i}.png'),imgs)
