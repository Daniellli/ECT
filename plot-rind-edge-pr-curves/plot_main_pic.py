'''
Author: xushaocong
Date: 2022-07-26 20:02:40
LastEditTime: 2022-07-27 10:55:09
LastEditors: xushaocong
Description: 
FilePath: /Cerberus-main/plot-rind-edge-pr-curves/plot_main_pic.py
email: xushaocong@stu.xmu.edu.cn
'''




import os
import os.path as osp
import glob
from loguru import logger
import torch 
from os.path import split , join 
import cv2

import numpy as np

from torchvision.transforms import transforms





import scipy.io as scio


def draw_all_testset():
    path = "/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/data/BSDS_RIND_mine/"
    normal = osp.join(path,"normal")
    reflectance = osp.join(path,"reflectance")
    depth = osp.join(path,"depth")
    illumination = osp.join(path,"illumination")
    image = osp.join(path,"images")

    target_path = osp.join("/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/plot-rind-edge-pr-curves/plot_material/tmp")


    images = sorted(glob.glob(image+"/*.jpg"))

    # RE_color=(255,192,0)
    # NE_color=(0,255,255)
    # IE_color=(14,253,0)
    # DE_color=(255,0,1)


    RE_color=(0,192,255)
    NE_color=(255,255,0)
    IE_color=(0,253,14)
    DE_color=(1,0,255)


    


    for idx, x in enumerate(images):
        
        name =   split(x)[-1].split('.')[-2]
        origin_img= cv2.imread(x)
        t_normal  = cv2.imread(osp.join(normal,name+".png"),cv2.IMREAD_GRAYSCALE) 
        t_reflectance  = cv2.imread(osp.join(reflectance,name+".png"),cv2.IMREAD_GRAYSCALE)
        t_depth  = cv2.imread(osp.join(depth,name+".png"),cv2.IMREAD_GRAYSCALE)
        t_illumination  = cv2.imread(osp.join(illumination,name+".png"),cv2.IMREAD_GRAYSCALE)

        O1=  origin_img.copy()
        O2=  origin_img.copy()
        O3=  origin_img.copy()
        O4=  origin_img.copy()
        
        
        O1[t_normal == 255] =NE_color
        O2[t_reflectance == 255] =RE_color
        O3[t_depth == 255] =DE_color
        O4[t_illumination == 255] =IE_color
        
        cv2.imwrite(osp.join(target_path,"normal_%04d.jpg"%(idx)),O1)
        cv2.imwrite(osp.join(target_path,"t_reflectance_%04d.jpg"%(idx)),O2)
        cv2.imwrite(osp.join(target_path,"t_depth_%04d.jpg"%(idx)),O3)
        cv2.imwrite(osp.join(target_path,"t_illumination_%04d.jpg"%(idx)),O4)
        



        
'''
description:  绘制 特定的一张测试集图像
param {*} image_path
return {*}
'''
def draw_specific_image(image_name):
    path = "/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/data/BSDS_RIND_mine/"
    normal = osp.join(path,"normal")
    reflectance = osp.join(path,"reflectance")
    depth = osp.join(path,"depth")
    illumination = osp.join(path,"illumination")
    image = osp.join(path,"images")

    target_path = osp.join("/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/plot-rind-edge-pr-curves/plot_material/tmp2")
    if not osp.exists(target_path):
        os.makedirs(target_path)





    RE_color=(0,192,255)
    NE_color=(255,255,0)
    IE_color=(0,253,14)
    DE_color=(1,0,255) 
    E_color=[244, 35, 232]
    

    idx=-1

    draw_image  = osp.join(image,image_name)
    name =   split(draw_image)[-1].split('.')[-2]
    origin_img= cv2.imread(draw_image)
    t_normal  = cv2.imread(osp.join(normal,name+".png"),cv2.IMREAD_GRAYSCALE) 
    t_reflectance  = cv2.imread(osp.join(reflectance,name+".png"),cv2.IMREAD_GRAYSCALE)
    t_depth  = cv2.imread(osp.join(depth,name+".png"),cv2.IMREAD_GRAYSCALE)
    t_illumination  = cv2.imread(osp.join(illumination,name+".png"),cv2.IMREAD_GRAYSCALE)

    t_edge = np.zeros(t_illumination.shape)
    t_edge[(t_illumination==255) | (t_normal==255)| (t_depth==255)| (t_reflectance==255)] = 255 

    # edge = scio.loadmat("/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/dataset/BSDS-RIND/testgt/edge/"+image_name.split('.')[-2]+".mat")
    # edge1 = edge["groundTruth"][0][0][0][0][0]



    
    O1=  origin_img.copy()
    O2=  origin_img.copy()
    O3=  origin_img.copy()
    O4=  origin_img.copy()
    O5=  origin_img.copy()
    
    
    
    O1[t_normal == 255] =NE_color
    
    O2[t_reflectance == 255] =RE_color
    O3[t_depth == 255] =DE_color
    O4[t_illumination == 255] =IE_color
    O5[t_edge == 255] =E_color
    
    
    cv2.imwrite(osp.join(target_path,"t_normal.jpg"),O1)
    cv2.imwrite(osp.join(target_path,"t_reflectance.jpg"),O2)
    cv2.imwrite(osp.join(target_path,"t_depth.jpg"),O3)
    cv2.imwrite(osp.join(target_path,"t_illumination.jpg"),O4)
    cv2.imwrite(osp.join(target_path,"t_edge.jpg"),O5)
        

    
    split_to_16_part(origin_img,target_path)


def split_to_16_part(img,target_path):
    H,W,C=img.shape
    
    tmp = img.copy()

    part_h = H//4
    part_w = W//4
    
    color=[255,255,255]


    #* 绘制横白线
    for i in range(1,4):
        tmp[part_h*i:(part_h*i+5),:] =  color


    #*
    for i in range(1,4):
        tmp[:,part_w*i:(part_w*i+5)] =  color
        
        

    cv2.imwrite(osp.join(target_path,"t_input.jpg"),tmp)
    
        

    logger.info("hello world")



if __name__=="__main__":
    draw_specific_image("102062.jpg")








