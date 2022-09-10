'''
Author: xushaocong
Date: 2022-09-06 13:40:43
LastEditTime: 2022-09-07 10:43:17
LastEditors: xushaocong
Description: 
FilePath: /Cerberus-main/plot/draw_inference.py
email: xushaocong@stu.xmu.edu.cn
'''

import os
import os.path as osp
import glob
from loguru import logger
import torch 
import cv2
import torch
from PIL import Image

from tqdm import tqdm


import time
import numpy as np 
import torch

from IPython import embed


from my_script.utils import interp_img

#* 每种边缘对应的颜色
RINDE_COLOR = [
    (10,139,226),
    (142,217,199),
    (235,191,114),
    (174, 125, 176),
    # (219, 118, 2)
]

TASKS = ["reflectance","illumination","normal","depth"]





'''
description:   给定图像已经对应的阈值, 根据阈值过滤edge 绘制图像, 
param {*} im_name 
param {*} threshold_list :rind sequence   for best ois threshold 
param {*} inference_res_path : 要找的图片的root, 可以在这个目录找到图像
param {*} save_dir: 结果保存路径
error_threashold : 低于这个阈值就直接 认为没有这类边缘了
return {*} A for RIND, B for edge 
'''
def draw_rind(im_path,inference_res_path,threshold= 0.5):
    im_name = im_path.split('/')[-1].split('.')[0]
    origin_im = cv2.imread(im_path)
    
    for  idx,k in enumerate(TASKS[::-1]):
        tmp_im = cv2.imread(osp.join(inference_res_path,k,'nms',im_name+'.png'),cv2.IMREAD_GRAYSCALE)#* the image coresponding to the RIND
        origin_im[tmp_im >threshold*255]= RINDE_COLOR[::-1][idx]
    return origin_im
        


'''
description:每个pixel 只取响应值最大的
param {*} im_path
param {*} inference_res_path
param {*} threshold
return {*}
'''
def draw_rind2(im_path,inference_res_path,threshold= 0.5):
    im_name = im_path.split('/')[-1].split('.')[0]
    origin_im = cv2.imread(im_path)
    
    #* read
    all_map = []
    for  idx,k in enumerate(TASKS):
        # tmp_im = 
        all_map.append(cv2.imread(osp.join(inference_res_path,k,'nms',im_name+'.png'),cv2.IMREAD_GRAYSCALE))#* the image coresponding to the RIND
    
    all_map = torch.from_numpy(np.array(all_map))
    
    copy_ = all_map.clone()
    value,indices= torch.max(copy_,0)
    
    #* 取最大的value 的type进行绘制 
    #* 再删除 低于0.5的
    H,W=indices.shape

    for i in range(H):
        for j in range(W):
            if value[i,j]> 255*threshold: #* 小于这个阈值认为是无效edge , 不进行绘制
                origin_im[i,j] = RINDE_COLOR[indices[i,j]]

    return origin_im
        




def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)



def imgs2video(img_root,video_name=None,fps=20  ):
    all_imgs = sorted(glob.glob(osp.join(img_root,"*")))
    demo  = cv2.imread(all_imgs[0])
    H,W,C=demo.shape 
    #保存视频的FPS，可以适当调整
    
    if video_name is None:
        video_name = time.strftime("%Y:%m:%d",time.gmtime(time.time()))+"_"+str(int(time.time()))+".avi"

    #可以用(*'DVIX')或(*'X264'),如果都不行先装ffmepg: sudo apt-get install ffmepg
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    videoWriter = cv2.VideoWriter(osp.join(img_root,'..',video_name),fourcc,fps,(W,H))#最后一个是保存图片的尺寸

    #for(i=1;i<471;++i)
    for idx,im in tqdm(enumerate(all_imgs)):
        frame = cv2.imread(im)
        videoWriter.write(frame)
    videoWriter.release()




def draw(origin_path,inference_res_path,origin_img_suffix='png'):

    save_path = osp.join(osp.dirname(origin_path),'inference_res')#* plot save path
    make_dir(save_path)
    all_img_names = sorted([x.split('.')[0] for x in os.listdir(origin_path) if x.endswith(f'.{origin_img_suffix}') ])
    for im_name in tqdm(all_img_names):
        img= draw_rind2(osp.join(origin_path,im_name+f'.{origin_img_suffix}'),inference_res_path,threshold=0.5)
        header = cv2.imread(osp.join(inference_res_path,'..','..','header.png'))
        H,_,_ = header.shape

        _,W,_=img.shape
        header = interp_img(header,(H//4,W))

        img = np.concatenate([header,img])
        
        cv2.imwrite(osp.join(save_path,im_name+".png"),img)
        
    imgs2video(save_path)

if __name__ == "__main__":

    origin_path = "/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/plot/demo/0001/imgs"
    inference_res_path = "/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/plot/demo/0001/2022:09:06_1662477068_0"
    draw(origin_path,inference_res_path,origin_img_suffix='png')


    






