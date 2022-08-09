'''
Author: xushaocong
Date: 2022-07-26 20:02:40
LastEditTime: 2022-08-09 18:57:48
LastEditors: xushaocong
Description: 
FilePath: /Cerberus-main/plot-rind-edge-pr-curves/plot_main_pic.py
email: xushaocong@stu.xmu.edu.cn
'''



from os import listdir
from PIL import Image
import time

from genericpath import exists
import os
import os.path as osp
import glob
from loguru import logger
import torch 
from os.path import split , join 
import cv2

import numpy as np

import skimage

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
description:  对goal 的非零元素进行扩张, 扩张倍数为10 
param {*} goal 
param {*} times
return {*}
'''
def dilation(goal, times = 2 ):
    selem = skimage.morphology.disk(times)


    goal = skimage.morphology.binary_dilation(goal, selem) != True
    goal = 1 - goal * 1.
    goal*=255
    
    return goal



        
'''
description:  绘制 特定的一张测试集图像
param {*} origin_image_pathath
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





    
    # RE_color=(255,192,255)
    # NE_color=(255,255,0)
    # IE_color=(0,253,14)
    # DE_color=(244, 35, 232) 

    RE_color=(10,139,226)
    NE_color=(235,191,114)
    IE_color=(142,217,199)
    DE_color=(174, 125, 176) 



    E_color=[219, 118, 2]#* BGR
    

    idx=-1

    draw_image  = osp.join(image,image_name)
    name =   split(draw_image)[-1].split('.')[-2]
    origin_img= cv2.imread(draw_image)

    t_normal  = cv2.imread(osp.join(normal,name+".png"),cv2.IMREAD_GRAYSCALE) 
    t_normal = dilation(t_normal)
    t_reflectance  = cv2.imread(osp.join(reflectance,name+".png"),cv2.IMREAD_GRAYSCALE)
    t_reflectance = dilation(t_reflectance)
    t_depth  = cv2.imread(osp.join(depth,name+".png"),cv2.IMREAD_GRAYSCALE)
    t_depth = dilation(t_depth)
    t_illumination  = cv2.imread(osp.join(illumination,name+".png"),cv2.IMREAD_GRAYSCALE)
    t_illumination = dilation(t_illumination)




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




'''
description: 
return {*}
'''
def draw_inverse_loss_conceptual_plot(
    model_res="/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/networks/precomputed/rindnet-resnet50",
    origin_image_path="/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/dataset/BSDS-RIND/test",
    save_dir = "/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/plot-rind-edge-pr-curves/plot_material/tmp",
    threshold=None,
    is_rindnet_res= True,
    is_concat_save=True,
):

    save_dir2=None
    if not is_concat_save:
        save_dir2 = save_dir+"_edge"
        

    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    if  save_dir2 is not None and not osp.exists(save_dir2):
        os.makedirs(save_dir2)
        

    if is_rindnet_res:
        task = ["reflectance","illumination","normal","depth","all_edges"]
    else :
        task = ["reflectance","illumination","normal","depth","edge"]

    
    threshold = np.array(threshold)*255
    
    RE_color=(10,139,226)
    NE_color=(235,191,114)
    IE_color=(142,217,199)
    DE_color=(174, 125, 176) 
    E_color=[219, 118, 2]#* BGR

    all_imgs= sorted(os.listdir(osp.join(model_res,task[0],'nms/')))
    for im in all_imgs:

        res = cv2.imread(osp.join(origin_image_path,im.replace('png','jpg')))
        res_2 = res.copy()

        maps = {}
        
        for t in task:
            maps[t]= cv2.imread(osp.join(model_res,t,"nms",im),cv2.IMREAD_GRAYSCALE)
        
        
     
        res[maps[task[0]] > threshold[0]] = RE_color #* 5976
        res[maps[task[1]] > threshold[1]] = IE_color
        res[maps[task[2]] > threshold[2]] = NE_color
        res[maps[task[3]] > threshold[3]] = DE_color
        res_2[maps[task[4]] > threshold[4]] = E_color


        if is_concat_save:
            cv2.imwrite(osp.join(save_dir,im),np.concatenate([res,res_2],axis=1))
        else :
            cv2.imwrite(osp.join(save_dir,im),res)
            cv2.imwrite(osp.join(save_dir2,im),res_2)






'''
description: 
return {*}
'''
def draw_inverse_loss_conceptual_plot_gt(
    model_res="/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/networks/precomputed/rindnet-resnet50",
    origin_image_path="/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/dataset/BSDS-RIND/test",
    save_dir = "/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/plot-rind-edge-pr-curves/plot_material/tmp",
    is_concat_save=False
):
    
    save_dir2=None
    if not is_concat_save:
        save_dir2 = save_dir+"_edge"

    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    if save_dir2 is not None and   not osp.exists(save_dir2):
        os.makedirs(save_dir2)

    
    task = ["reflectance","illumination","normal","depth"]


    all_imgs= sorted([x for x in os.listdir(origin_image_path) if x.endswith('jpg')])
    
    for im in all_imgs:

        RE_color=(10,139,226)
        NE_color=(235,191,114)
        IE_color=(142,217,199)
        DE_color=(174, 125, 176) 


        E_color=[219, 118, 2]#* BGR

        maps = {}
        for t in task:
            maps[t]= cv2.imread(osp.join(model_res,t,im.replace('jpg','png')),cv2.IMREAD_GRAYSCALE)
        
        
        res = cv2.imread(osp.join(origin_image_path,im))
        
        res_2 = res.copy()
        res[maps[task[0]] ==255] = RE_color #* 5976
        res[maps[task[1]] ==255] = IE_color
        res[maps[task[2]] ==255] = NE_color
        res[maps[task[3]] ==255] = DE_color

        
        res_2[((maps[task[0]] ==255 ) | (maps[task[1]] ==255 ) | (maps[task[2]] ==255 ) | (maps[task[3]] ==255 ) )] = E_color

        if is_concat_save:
            cv2.imwrite(osp.join(save_dir,im),np.concatenate([res,res_2],axis=1))
        else :
            cv2.imwrite(osp.join(save_dir,im),res)
            cv2.imwrite(osp.join(save_dir2,im),res_2)



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
 

'''
description:  给定两个生成的结果路径,两个路径的结果一一对应,然后将其concat在一起存储到concat_save_path
return {*}
'''
def concat_for_comparison(
    without_loss_save_path,
    gt_save_dir,
    concat_save_path = "/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/plot-rind-edge-pr-curves/plot_material/concat_loss_compare2"
    ):
      
    #* 读取然后concat在一起
    
    

    if not osp.exists(concat_save_path):
        os.makedirs(concat_save_path)

    for a,b in zip(sorted(glob.glob(osp.join(without_loss_save_path,'*.png'))),
                    sorted(glob.glob(osp.join(gt_save_dir,'*.jpg')))):
        
    #     #* 上面是没有constraint loss
    #     #* 下面是有constraint loss
        pinjie([Image.open(a),Image.open(b)],osp.join(concat_save_path,a.split('/')[-1].split('.')[0]+".png") )
        

if __name__=="__main__":
    # draw_specific_image("102062.jpg")
    with_loss_path="/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/networks/edge_cerberus8/edge_final_3_3090pth2_0"
    with_loss_save_path = "/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/plot-rind-edge-pr-curves/plot_material/with_loss_path"

    gt="/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/dataset/BSDS_RIND_mine"
    gt_save_dir = "/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/plot-rind-edge-pr-curves/plot_material/gt_plot_compara" 
    

    without_loss_path="/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/networks/edge_cerberus8/edge_without_constraint_losspth2_0"
    without_loss_save_path = "/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/plot-rind-edge-pr-curves/plot_material/without_loss_path"


    #* 绘制并写入
    tic = time.time()
    # threshold=[0.84,0.61,0.84,0.66,0.5]
    threshold=[0.918081,0.93,0.88,0.82,0.76] #* for 8068
    draw_inverse_loss_conceptual_plot(without_loss_path,
    save_dir=without_loss_save_path,
    is_rindnet_res=False,
    is_concat_save=False,
    threshold=threshold
    )
    logger.info(time.strftime("%H:%M:%S",time.gmtime(time.time()-tic)))

    # tic = time.time()
    # threshold=[0.48,0.94,0.01,0.13,0.18] #* for 35028, 0.01 有问题
    # threshold=[0.78,0.92,0.65,0.82,0.44] #* for 16068
    # threshold=[0.85,0.73,0.86,0.72,0.815859] #* for 302022
    # draw_inverse_loss_conceptual_plot(with_loss_path,
    # save_dir=with_loss_save_path,
    # is_rindnet_res=False,is_concat_save=True,
    # threshold=threshold
    # )

    # spend_time= time.time()-tic
    # logger.info(time.strftime("%H:%M:%S",time.gmtime(spend_time)))
    draw_inverse_loss_conceptual_plot_gt(gt,save_dir=gt_save_dir,is_concat_save=False)


    # tic = time.time()
    # concat_for_comparison(
    #     without_loss_save_path,
    #     gt_save_dir)
    # logger.info(time.strftime("%H:%M:%S",time.gmtime(time.time()-tic)))






 
