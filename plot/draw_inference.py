'''
Author: xushaocong
Date: 2022-09-06 13:40:43
LastEditTime: 2022-09-23 13:44:26
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
description:  读取所有的nms 结果map
param {*} im_path
param {*} nms_path
param {*} suffix
return {*}
'''
def read_rind_nms_img(im_path,nms_path,suffix = ".png"):
    im_name = im_path.split('/')[-1].split('.')[0]
    all_map = []
    for  idx,k in enumerate(TASKS):
        # tmp_im = 
        all_map.append(cv2.imread(osp.join(nms_path,k,'nms',im_name+suffix),cv2.IMREAD_GRAYSCALE))#* the image coresponding to the RIND
    return all_map
    

'''
description:每个pixel 只取响应值最大的
param {*} im_path
param {*} inference_res_path
param {*} threshold
return {*}
'''
def draw_rind2(im_path,inference_res_path,threshold= 0.5):
    origin_im = cv2.imread(im_path)
    #* read
    all_map = read_rind_nms_img(im_path,inference_res_path)
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
        video_name = time.strftime("%Y:%m:%d",time.gmtime(time.time()))+"_"+str(int(time.time()))+".mp4"

    #可以用(*'DVIX')或(*'X264'),如果都不行先装ffmepg: sudo apt-get install ffmepg
    # fourcc = cv2.VideoWriter_fourcc(*'XVID') #* 对应后缀: avi
    # fourcc = cv2.VideoWriter_fourcc(*'DVIX')
    # fourcc = cv2.VideoWriter_fourcc(*'X264')
    fourcc = cv2.VideoWriter_fourcc(*'MP4V') #* for mp4
    

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




'''
description:  upsample the matric in numpy array format  
param {*} data
return {*}
'''
def upsample_numpy_format(data,scale_factor =2 ):
    data = torch.from_numpy(data).permute([2,0,1])
    
    data2 = torch.nn.functional.interpolate(data.unsqueeze(0),scale_factor=scale_factor)
    return data2.squeeze().permute([1,2,0]).numpy()



'''
description:  过滤掉RIND中不是最大值的数据
param {*} rind_nms_map
return {*}
'''
def filter_by_maximun(rind_nms_map):

    x = torch.from_numpy(np.array(rind_nms_map))
    values,indices = torch.max(x,0)
    #* filter 
    rind_nms_map_filtered = []
    for idx,y in enumerate(rind_nms_map):
        tmp = np.zeros_like(y)
        tmp[indices==idx] = y[indices==idx] 
        
        rind_nms_map_filtered.append(tmp)
    return rind_nms_map_filtered
    

'''
description:  分别绘制每一个task的image 
param {*} origin_path
param {*} inference_res_path
param {*} origin_img_suffix
param {*} model_res_right : true的话, model结果放在原图右边, false就放在下方
return {*}
'''
def draw_grid(origin_path,inference_res_path,origin_img_suffix='png',model_res_right=False):
    #* 1. read origin image and RIND nms map 
    #* 2. upsample origin image for 2 times , 
    #* 3. concatenate all the results   as a grip, first colunm for origin image and second and third for RIND , format as 2 X 2 matric 
    #? how to upsampel  ? ----> interpolate  

    save_path = osp.join(osp.dirname(origin_path),'inference_res')#* plot save path
    make_dir(save_path)
    all_img_names = sorted([x.split('.')[0] for x in os.listdir(origin_path) if x.endswith(f'.{origin_img_suffix}') ])
    


    for im_name in tqdm(all_img_names):

        origin_img_path = osp.join(origin_path,im_name+f'.{origin_img_suffix}')
        
        origin_img = cv2.imread(origin_img_path)
        origin_img_max= upsample_numpy_format(origin_img)
        origin_img_max = cv2.cvtColor(origin_img_max,cv2.COLOR_RGB2BGR)
        origin_img_max = cv2.cvtColor(origin_img_max,cv2.COLOR_RGB2BGR)
        
        cv2.putText(origin_img_max,"Origin",(0,30),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)

        rind_nms_map=read_rind_nms_img(origin_img_path,inference_res_path)
        
        #* draw  
        for idx,x in enumerate(rind_nms_map):
            cv2.putText(x,TASKS[idx],(0,30),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)

        a = np.concatenate([rind_nms_map[0],rind_nms_map[1]],axis=1) #* reflectance , illumination
        b = np.concatenate([rind_nms_map[2],rind_nms_map[3]],axis=1) #* normal  , depth
        c = np.concatenate([a,b])

        if model_res_right:
            cv2.imwrite(osp.join(save_path,im_name+".png"),np.concatenate([origin_img_max,np.stack([c , c , c] , axis = 2)],axis=1))
        else :
            cv2.imwrite(osp.join(save_path,im_name+".png"),np.concatenate([origin_img_max,np.stack([c , c , c] , axis = 2)],axis=0))
    
        
    imgs2video(save_path)

'''
description:  分别绘制每一个task的image , RIND map只保留最大值
param {*} origin_path
param {*} inference_res_path
param {*} origin_img_suffix
return {*}
'''
def draw_grid2(origin_path,inference_res_path,origin_img_suffix='png',model_res_right=False):
    #* 1. read origin image and RIND nms map 
    #* 2. upsample origin image for 2 times , 
    #* 3. concatenate all the results   as a grip, first colunm for origin image and second and third for RIND , format as 2 X 2 matric 
    #? how to upsampel  ? ----> interpolate  

    save_path = osp.join(osp.dirname(origin_path),'inference_res')#* plot save path
    make_dir(save_path)
    # all_img_names = sorted([x.split('.')[0] for x in os.listdir(origin_path) if x.endswith(f'.{origin_img_suffix}') ])
    all_img_names = sorted(['.'.join(x.split('.')[0:2]) for x in os.listdir(origin_path) if x.endswith(f'.{origin_img_suffix}') ],key=lambda x:(x.split('.')[0],x.split('.')[1]))

    if len(os.listdir(save_path))  != len(all_img_names):
        for im_name in tqdm(all_img_names):

            origin_img_path = osp.join(origin_path,im_name+f'.{origin_img_suffix}')
            
            origin_img = cv2.imread(origin_img_path)
            origin_img_max= upsample_numpy_format(origin_img)

            origin_img_max = cv2.cvtColor(origin_img_max,cv2.COLOR_RGB2BGR)
            origin_img_max = cv2.cvtColor(origin_img_max,cv2.COLOR_RGB2BGR)
            
            cv2.putText(origin_img_max,"Origin",(0,30),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            
        
            rind_nms_map=read_rind_nms_img(origin_img_path,inference_res_path)

            rind_nms_map_filtered = filter_by_maximun(rind_nms_map)

            #* draw  
            for idx,x in enumerate(rind_nms_map_filtered):
                cv2.putText(x,TASKS[idx],(0,30),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)

            a = np.concatenate([rind_nms_map_filtered[0],rind_nms_map_filtered[1]],axis=1) #* reflectance , illumination
            b = np.concatenate([rind_nms_map_filtered[2],rind_nms_map_filtered[3]],axis=1) #* normal  , depth
            c = np.concatenate([a,b])


            if model_res_right:
                cv2.imwrite(osp.join(save_path,im_name+".png"),np.concatenate([origin_img_max,np.stack([c , c , c] , axis = 2)],axis=1))
            else :
                cv2.imwrite(osp.join(save_path,im_name+".png"),np.concatenate([origin_img_max,np.stack([c , c , c] , axis = 2)],axis=0))

    

    imgs2video(save_path,fps=30)




if __name__ == "__main__":


    BASE_PATH = "/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/plot/demo"


        
    # origin_path = osp.join(BASE_PATH,"demo3/Car3")
    # inference_res_path = osp.join(BASE_PATH,"demo3/nms_res_0")
    # draw_grid(origin_path,inference_res_path,origin_img_suffix='jpg',model_res_right=True)


    # origin_path = osp.join(BASE_PATH,"0000/imgs")
    # inference_res_path = osp.join(BASE_PATH,"0000/nms_res_0")
    # draw_grid2(origin_path,inference_res_path,origin_img_suffix='png',model_res_right=True)


    # origin_path = osp.join(BASE_PATH,"KITTI/imgs")
    # inference_res_path = osp.join(BASE_PATH,"KITTI/nms_res_0")
    # draw_grid2(origin_path,inference_res_path,origin_img_suffix='png')



    origin_path = osp.join(BASE_PATH,"robotics/imgs")
    inference_res_path = osp.join(BASE_PATH,"robotics/nms_res_0")
    draw_grid2(origin_path,inference_res_path,origin_img_suffix='png')


    


    






