'''
Author: xushaocong
Date: 2022-07-26 20:02:40
LastEditTime: 2022-08-28 23:49:52
LastEditors: xushaocong
Description: 
FilePath: /Cerberus-main/plot-rind-edge-pr-curves/plot_main_pic.py
email: xushaocong@stu.xmu.edu.cn
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
import skimage
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import scipy.io as scio
from PIL import Image
import json
import shutil


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
description:  绘制ppt的图像 
param {*} origin_image_pathath
return {*}
'''
def draw_specific_image(image_name,target_path):
    path = "/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/data/BSDS_RIND_mine/"
    normal = osp.join(path,"normal")
    reflectance = osp.join(path,"reflectance")
    depth = osp.join(path,"depth")
    illumination = osp.join(path,"illumination")
    image = osp.join(path,"images")


    RE_color=(10,139,226)
    NE_color=(235,191,114)
    IE_color=(142,217,199)
    DE_color=(174, 125, 176) 
    E_color=[219, 118, 2]#* BGR
    

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
        
    split_to_16_part(origin_img,osp.join(target_path,'split.png'))


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
        
    
    make_dir()
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
description:  将算法结果沿第一维度求最大值融合成一个tensor  , 
return {*}
'''
def maximun_for_contraint_loss(model_res,
                            save_dir = "/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/plot-rind-edge-pr-curves/plot_material/tmp",
):

    save_dir = osp.join(save_dir,'heat_map_3')
    if not  osp.exists(save_dir):
        os.makedirs(save_dir)
    
    task = ["reflectance","illumination","normal","depth"]
    

    all_imgs= sorted(os.listdir(osp.join(model_res,task[0],'nms/')))
    for idx,im in enumerate(all_imgs):
        

        tmp = []
        for t in task:
            tmp.append(cv2.imread(osp.join(model_res,t,"nms",im),cv2.IMREAD_GRAYSCALE))
        tmp = torch.from_numpy(np.array(tmp))
        rind,_ = torch.max(tmp,0)
        
        #!+==============
        edge = cv2.imread(osp.join(model_res,'edge',"nms",im),cv2.IMREAD_GRAYSCALE)
        edge  = torch.tensor(edge)
        rind = torch.cat([edge,rind],1)
        #!+==============
        
        #* 默认是6.4, 4.8 对应size 是640X480,  
        
        # plt.figure(figsize=(130,50))
        s = (np.array(rind.shape)/100).astype(np.float)
        
        plt.figure(figsize=(s[1],s[0]))
        
        plt.pcolor(rind, cmap='jet')
        # plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
        
        # plt.savefig('tmp.jpg',bbox_inches = 'tight', pad_inches = 0)
        #* 将 plt figure数据转numpy 矩阵
        canvas = FigureCanvasAgg(plt.gcf())
        canvas.draw()
        img = np.array(canvas.renderer.buffer_rgba())
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        
    
        # edge_ = cv2.imread(osp.join(model_res,'edge',"nms",im),cv2.IMREAD_GRAYSCALE)
        # Image.open(osp.join(model_res,'edge',"nms",im))

        #* 将numpy矩阵转 Pillow数据
        PIL_img = Image.fromarray(img) #data二维图片矩阵。
        PIL_img = PIL_img.rotate(180, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
        PIL_img.save(osp.join(save_dir,im))
        # PIL_img.save('tmp.jpg')
        
        

'''
description: 
return {*}
'''
def draw_inverse_loss_conceptual_plot_gt(
    model_res="/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/networks/precomputed/rindnet-resnet50",
    origin_image_path="/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/dataset/BSDS-RIND/test",
    save_dir = "/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/plot-rind-edge-pr-curves/plot_material/tmp",
    is_concat_save=False,
    rind_split_save = False
):
    
    save_dir2=None
    if not is_concat_save:
        save_dir2 = save_dir+"_edge"

    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    if save_dir2 is not None and   not osp.exists(save_dir2):
        os.makedirs(save_dir2)

    
    task = ["reflectance","illumination","normal","depth"]

    #* 读取所以图像了, 因为是gt
    all_imgs= sorted([x for x in os.listdir(osp.join(model_res,'images')) if x.endswith('jpg')])
    
    RE_color=(10,139,226)
    NE_color=(235,191,114)
    IE_color=(142,217,199)
    DE_color=(174, 125, 176) 


    E_color=[219, 118, 2]#* BGR

    #* test image==============

    test1 = cv2.imread(osp.join(model_res,'images',all_imgs[0]))
    test2 = test1.copy()
    test3 = test1.copy()
    test4 = test1.copy()
    test5 = test1.copy()
    test6 = test1.copy()
    
    test1[:] = RE_color
    test2[:] = NE_color
    test3[:] = IE_color
    test4[:] = DE_color
    test5[:] = E_color
    test6[:]=(0,0,255)

    cv2.imwrite(osp.join(save_dir,'r.jpg'),test1)
    cv2.imwrite(osp.join(save_dir,'n.jpg'),test2)
    cv2.imwrite(osp.join(save_dir,'i.jpg'),test3)
    cv2.imwrite(osp.join(save_dir,'d.jpg'),test4)
    cv2.imwrite(osp.join(save_dir,'e.jpg'),test5)
    cv2.imwrite(osp.join(save_dir,'e.jpg'),test5)
    cv2.imwrite(osp.join(save_dir,'composition.jpg'),test6)
    

    #*========================

    reflec_path = osp.join(osp.dirname(save_dir),'reflectance')
    
    make_dir(reflec_path)

    illu_path = osp.join(osp.dirname(save_dir),'illumination')
    make_dir(illu_path)

    normal_path = osp.join(osp.dirname(save_dir),'normal')
    make_dir(normal_path)

    depth_path = osp.join(osp.dirname(save_dir),'depth')
    make_dir(depth_path)
    edge_path = osp.join(osp.dirname(save_dir),'edge')
    make_dir(edge_path)


    for im in all_imgs:
        maps = {}
        for t in task:
            # maps[t]=dilation( cv2.imread(osp.join(model_res,t,im.replace('jpg','png')),cv2.IMREAD_GRAYSCALE))
            maps[t]=dilation( cv2.imread(osp.join(model_res,t,im.replace('jpg','png')),cv2.IMREAD_GRAYSCALE),times=1.5)
            # maps[t]= cv2.imread(osp.join(model_res,t,im.replace('jpg','png')),cv2.IMREAD_GRAYSCALE)
        res = cv2.imread(osp.join(model_res,'images',im))
        
        
        res_2 = res.copy()

        if rind_split_save:
            # r_res = res.copy()
            # i_res = res.copy()
            # n_res = res.copy()
            # d_res = res.copy()
            e_res = res.copy()

            # r_res[maps[task[0]] ==255] = (0,0,255) #* 5976
            # i_res[maps[task[1]] ==255] = (0,0,255)
            # n_res[maps[task[2]] ==255] = (0,0,255)
            # d_res[maps[task[3]] ==255] = (0,0,255)

            e_res[((maps[task[0]] ==255 ) | (maps[task[1]] ==255 ) | (maps[task[2]] ==255 ) | (maps[task[3]] ==255 ) )] = (0,0,255)

            # cv2.imwrite(osp.join(reflec_path,im.replace('jpg','png')),r_res)
            # cv2.imwrite(osp.join(illu_path,im.replace('jpg','png')),i_res)
            # cv2.imwrite(osp.join(normal_path,im.replace('jpg','png')),n_res)
            # cv2.imwrite(osp.join(depth_path,im.replace('jpg','png')),d_res)
            cv2.imwrite(osp.join(edge_path,im.replace('jpg','png')),e_res)
            
            continue        
        res[maps[task[0]] ==255] = RE_color #* 5976
        res[maps[task[1]] ==255] = IE_color
        res[maps[task[2]] ==255] = NE_color
        res[maps[task[3]] ==255] = DE_color


        res[((maps[task[0]] ==255 ) &  (maps[task[1]] ==255 ) & (maps[task[2]] ==255 ) & (maps[task[3]] ==255 ) )] = (0,0,255)
        res_2[((maps[task[0]] ==255 ) | (maps[task[1]] ==255 ) | (maps[task[2]] ==255 ) | (maps[task[3]] ==255 ) )] = E_color


        if is_concat_save:
            cv2.imwrite(osp.join(save_dir,im.replace('jpg','png')),np.concatenate([res,res_2],axis=1))
        else :
            cv2.imwrite(osp.join(save_dir,im.replace('jpg','png')),res)
            cv2.imwrite(osp.join(save_dir2,im.replace('jpg','png')),res_2)



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
        


'''
description:  concat  两个路径下的图像 , 为了合成with loss and without loss heat map 方便对比
param {*} with_loss_save_path
param {*} without_loss_save_path
return {*}
'''
def concat_two_set(with_loss_save_path,without_loss_save_path):

    path_a = osp.join(with_loss_save_path,'heat_map_3')
    path_b = osp.join(without_loss_save_path,'heat_map_3')

    save_path = "/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/plot-rind-edge-pr-curves/plot_material/with_and_without_loss_compare"
    if not osp.exists(save_path):
        os.makedirs(save_path)
    all_ims = sorted(os.listdir(path_a))

    for i in all_ims:
        
        x= cv2.imread(osp.join(path_a,i))
        y= cv2.imread(osp.join(path_b,i))
        #*  row 1 : with loss ,  
        #*  row 2 : without loss 
        #* col 1:  edge
        #* col 2:  rind
        
        cv2.putText(x,"with loss,edge                  rind", (0, 30), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 3)
        cv2.putText(y,"without loss,edge                  rind", (0, 30), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 3)

        cv2.imwrite(osp.join(save_path,i) ,np.concatenate([x,y]))

        

'''
description:  concatenate two image 
param {*} imgA : 有loss的
param {*} imgB: 无loss的
param {*} save_file
return {*}
'''
def concat_two_img(imgA,imgB,save_file,):
    x = cv2.imread(imgA)
    y = cv2.imread(imgB)
     
    cv2.putText(x,"Ours ,edge                  rind", (0, 30), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 3)
    cv2.putText(y,"RINDNET loss,edge                  rind", (0, 30), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 3)

    
    
    cv2.imwrite(save_file ,np.concatenate([x,np.ones([20,x.shape[1],3])*255,y]))





'''
description:   给定图像已经对应的阈值, 根据阈值过滤edge 绘制图像, 
param {*} im_name 
param {*} threshold_list :rind sequence   for best ois threshold 
param {*} img_path_root : 要找的图片的root, 可以在这个目录找到图像
param {*} save_dir: 结果保存路径
error_threashold : 低于这个阈值就直接 认为没有这类边缘了
return {*} A for RIND, B for edge 
'''
def draw_rind_on_img_with_threshold(im_name,threshold_list,img_path_root,save_dir,error_threashold = 0.1):

    RE_color=(10,139,226)
    IE_color=(142,217,199)
    NE_color=(235,191,114)
    DE_color=(174, 125, 176) 
    E_color=[219, 118, 2]#* BGR

    origin_im = cv2.imread(osp.join(ORIGIN_IMG_PATH,im_name+'.jpg'))
    

    #* save edge =============================================    
    origin_im2 = origin_im.copy()
    egde = cv2.imread(osp.join(img_path_root,'edge','nms',im_name+'.png'),cv2.IMREAD_GRAYSCALE)
    origin_im2[egde>255*threshold_list['edge']]  = E_color
    res_name_edge = osp.join(save_dir,"%s_compare_loss_edge.png"%(im_name))
    cv2.imwrite(res_name_edge,origin_im2)
    #* =======================================================
    
    # origin_im[egde >v*255]= RINDE_COLOR[idx]
    for  idx,(k,v) in enumerate(threshold_list.items()):
        #* judge edge 
        if idx== 4:
            break
        tmp_im = cv2.imread(osp.join(img_path_root,k,'nms',im_name+'.png'),cv2.IMREAD_GRAYSCALE)#* the image coresponding to the RIND

        if v < error_threashold :
            origin_im[tmp_im >0.6*255]= RINDE_COLOR[idx]
        else :
            origin_im[tmp_im >v*255]= RINDE_COLOR[idx]
        

        #* 扩大后没法看, 都挤在一起了
        # tmp_im[tmp_im<=v*255] = 0
        # tmp_im = dilation(tmp_im,1)
        # origin_im[tmp_im>v*255]= RINDE_COLOR[idx]

    
    res_name = osp.join(save_dir,"%s_compare_loss.png"%(im_name))
    cv2.imwrite(res_name,origin_im)


    res_name_concat = osp.join(save_dir,"%s_compare_loss_concat.png"%(im_name))

    concat_res = np.concatenate([origin_im,np.ones([origin_im.shape[0],10,3])*255,origin_im2],axis=1)
    cv2.imwrite(res_name_concat,concat_res)

    return res_name,res_name_edge,res_name_concat
    

    


def load_json(path):


    with open(path,'r') as f:

        data = json.load(f)

    return data



def make_dir(path):
    if  not osp.exists(path):
        os.makedirs(path)



'''
description:  delete  the  file with the  corresponding suffix
param {*} dir_path
param {*} suffix
return {*}
'''
def delete_file_with_suffix(dir_path,suffix):

    
    all_file = glob.glob(dir_path+"/*."+suffix)

    for f in all_file:
        os.remove(f)

'''
description:  读取json 文件,读取对比的结果,
 将对比的结果分别先根据对比的效果比较好的几个sample根据ois绘制RIND和edge,
 然后分别保存到a_save_path和b_save_path, 然后最后将 两个路径的结果concat到一起存储到save_path
param: compara_json_path : 包含对比的实验结果的json 文件

return {*}
'''
def draw_topK_result(compara_json_path,a_path,a_save_path,
                    b_path,b_save_path,save_path):

    make_dir(a_save_path)
    make_dir(b_save_path)
    make_dir(save_path)
    

    compare_data = load_json(compara_json_path)
    for im_name,value in compare_data.items():
        a_threshold= {}
        b_threshold= {}
        for k,v in value['A1'].items():
            a_threshold[k] = float(v[1])#* pick threshold 

        for k,v in value['B1'].items():
            b_threshold[k] = float(v[1])#* pick threshold 
            
        #* return file name 
        rind_a,edge_a,concat_a = draw_rind_on_img_with_threshold(im_name,a_threshold,a_path,a_save_path)#* A1 是带有loss的
        rind_b,edge_b,concat_b= draw_rind_on_img_with_threshold(im_name,b_threshold,b_path,b_save_path)#* 
        concat_two_img(concat_a,concat_b,osp.join(save_path,im_name+'_final_compare.png'))
        print(a_threshold,b_threshold)
            


'''
description:  绘制loss 概念图的相关代码
return {*}
'''
def tmp():
   pass
    #* 绘制并写入
    # tic = time.time()
    # threshold=[0.84,0.61,0.84,0.66,0.5]
    # threshold=[0.918081,0.93,0.88,0.82,0.76] #* for 8068
    # draw_inverse_loss_conceptual_plot(
    #     without_loss_path,
    #     save_dir=without_loss_save_path,
    #     is_rindnet_res=False,
    #     is_concat_save=False,
    #     threshold=threshold
    # )
    # logger.info(time.strftime("%H:%M:%S",time.gmtime(time.time()-tic)))


'''
description:  两个mat格式的gt和 png 格式的gt 是否一致
return {*}
'''
# def check_consistence():
#     a = scio.loadmat(osp.join(ORIGIN_IMG_GT,'depth/2018.mat'))
#     b = cv2.imread(osp.join(ORIGIN_IMG_GT2,'depth/2018.png'),cv2.IMREAD_GRAYSCALE)
#     a = a['groundTruth'][0][0][0][0][0] 
#     if (b==255).sum()==(a==255).sum():
#         logger.info("yes")
    
            



def save_path(img,path):
    if not osp.exists(osp.dirname(path)):
        os.makedirs(osp.dirname(path))

    cv2.imwrite(path,img)




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
    
    


'''
description:  根据 performance 的图像以及对应的ois绘制图像, 
param {*} path : performance 评估的图像存储路径, 从这里读取nms后的图像
param {*} tasks: all task
param {*} save_path : 结果保存的路径
return {*}
'''
def draw_precision_according_best_performace(path,tasks,save_p):

    make_dir(save_p)

    performance = get_best_ois_for_each_task(path,tasks)

    for idx, (k,v) in enumerate(performance.items()):
        
        image_name, threshold=v
        origin_img = cv2.imread(osp.join(ORIGIN_IMG_GT2,'images',image_name+".jpg"))

        #* get image mask  
        pred_res = cv2.imread(osp.join(path,k,'nms',image_name+".png"),cv2.IMREAD_GRAYSCALE)
        pred_mask = pred_res >(threshold*255)   #* filter by ois threshold 
        
        #* draw
        draw_threshold_image(origin_img,pred_mask,image_name,k)

        #* save 
        save_path(origin_img,osp.join(save_p,f'{image_name}_{k}.png'))





'''
description:   将单个图像 单个task 的多个ois绘制出来
param {*} path : 多个算法评估结果文件
param {*} thresholds : 多个算法评估结果在image_name这张图像上的ois阈值
param {*} image_name : 要绘制的图像
param {*} save_path : 结果保存的路径,每个任务都不一样
return {*}
'''
def draw_precision(paths,task,image_name,save_paths,threshold=0.5):

    
    src_img = cv2.imread(osp.join(ORIGIN_IMG_GT2,'images',image_name+".jpg"))
    for path,save_p in zip(paths,save_paths):
        make_dir(save_p)
        origin_img= src_img.copy()
    
        #* get image mask  
        pred_res = cv2.imread(osp.join(path,task,'nms',image_name+".png"),cv2.IMREAD_GRAYSCALE)
        pred_mask = pred_res >=(threshold *255)  #* filter by ois threshold 
        # pred_mask_dilate =  dilation(pred_mask,1.2)==255
        
        #* draw
        origin_img = draw_F_image(origin_img,pred_mask,image_name,task)
   
        # origin_img = draw_F_image(origin_img,pred_mask,pred_mask_dilate,image_name,task)

        #* save
        save_path(origin_img,osp.join(save_p,f'{image_name}_{task}.png'))



'''
description:  统计像素的响应值 
param {*} pred_res :要统计的map
return {*}
'''
def statistic_edge_pixel(pred_res):
    statistic = {}
    range_ = [0,5,10,20,30,40,50,100,150,200,255]
    for i in range(len(range_)-1):
        statistic[f"{range_[i]} to {range_[i+1]}"] =((range_[i+1]>pred_res )  & (range_[i]<pred_res ) ).sum()
    # print(f"image_name : {image_name} , path {path} , task : {task}")
    for k,v in statistic.items():
        logger.info(f"{k} : {v}")




'''
description: 读取gt  绘制TP,FP,FN图像,dilate GT
param {*} image : 图像,  
param {*} pred_mask : 预测的mask
param {*} image_name : 图像名字
param {*} task_name : 任务名字
return {*}
'''
def draw_F_image(image,pred_mask,image_name,task_name):

   

    #* GT MASK
    #* GT 读不了  edge  MASK
    if task_name== "all_edges":
        # gt_mask = read_mat_gt(osp.join(ORIGIN_IMG_GT,k,image_name+".mat"))#* 无法读取edge 
        gt_mask=np.zeros(pred_mask.shape,dtype=bool)
        for t in TASKS[:-1]:
            gt_mask = (gt_mask | ( cv2.imread(osp.join(ORIGIN_IMG_GT2,t,image_name+".png"),cv2.IMREAD_GRAYSCALE)==255))
    else :
        gt_mask = cv2.imread(osp.join(ORIGIN_IMG_GT2,task_name,image_name+".png"),cv2.IMREAD_GRAYSCALE)==255
    #!================
    gt_mask_dilate = dilation(gt_mask,times=1.2)==255
    #!================
    
    #* draw TP,FN,FP,
    image[(gt_mask==False) & pred_mask ] = COLORS['FP'] #* False positive 
    image[gt_mask & (pred_mask == False)] = COLORS['FN'] #* False negtive
    image[gt_mask_dilate & pred_mask] = COLORS['TP'] #* true positive  
    return image






'''
description: 
param {*} img
param {*} task_name
return {*}
'''
def draw_gt(image_name,task_name):

    image = cv2.imread(osp.join(ORIGIN_IMG_GT2,'images',image_name+".jpg"))
    #* GT MASK
    #* GT 读不了  edge  MASK
    if task_name== "all_edges":
        gt_mask=np.zeros(image.shape[:2],dtype=bool)
        for t in TASKS[:-1]:
            gt_mask = (gt_mask | ( cv2.imread(osp.join(ORIGIN_IMG_GT2,t,image_name+".png"),cv2.IMREAD_GRAYSCALE)==255))
    else :
        gt_mask = cv2.imread(osp.join(ORIGIN_IMG_GT2,task_name,image_name+".png"),cv2.IMREAD_GRAYSCALE)==255

    gt_mask = dilation(gt_mask,times=1.2)==255

    #* draw TP,FN,FP,
    image[gt_mask]= COLORS['TP'] #* true positive  
    return image


'''
description:  删除一个目录list下的所有文件和目录
param {*} dir_list
return {*}
'''
def   delete_dir(dir_list):
    for p in dir_list:
        shutil.rmtree(p, ignore_errors=False, onerror=None)

'''
description:  绘制gt的map ,  用于绘制qualitative result用
param {*} all_images
param {*} gt_save_path
return {*}
'''
def draw_gt_map(all_images,gt_save_path):

    for image_name in all_images:
        for task in TASKS:
            image = draw_gt(image_name,task)
            save_path(image,osp.join(gt_save_path,f'{image_name}_{task}.png'))




'''
description:  将多个算法结果可视化 的结果 concat 在一起
param {*} all_images
param {*} src_paths
param {*} save_path
return {*}
'''
def draw_compare_plot(all_images,src_paths,save_path):

    for image_name in all_images:
        for task in TASKS:
            imgs = None
            for idx, model_save_path in enumerate(src_paths):
                if idx == 0:
                    imgs = cv2.imread(osp.join(model_save_path,f'{image_name}_{task}.png'))
                    imgs = np.concatenate([imgs,np.ones([imgs.shape[0],10,3])*255],1)
                else :
                    imgs = np.concatenate([imgs,
                                    cv2.imread(osp.join(model_save_path,f'{image_name}_{task}.png')),
                                    np.ones([imgs.shape[0],10,3])*255],1)
            cv2.imwrite(osp.join(save_path, f'{image_name}_{task}.png'),imgs)


if __name__=="__main__":


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

    # draw_specific_image("102062.jpg")
    edge_cerberus= osp.join(EVAL_RES_ROOT,"final_version/edge_final_8_3090_0")
    edge_cerberus_save_path =  osp.join(SAVE_ROOT,"Ours")
    
    # without_loss_path= osp.join(EVAL_RES_ROOT,"edge_cerberus8/edge_without_constraint_losspth2_0")
    without_loss_path= osp.join(EVAL_RES_ROOT,"final_version/edge_final_4_A100_80G_no_loss_0")
    without_loss_save_path =osp.join(SAVE_ROOT,"without_loss_path")
    

    RINDNET_path= osp.join(EVAL_RES_ROOT,"precomputed/rindnet-resnet50")
    RINDNET_save_path = osp.join(SAVE_ROOT,"RINDNet")

    DFF_path= osp.join(EVAL_RES_ROOT,"precomputed/dff")
    DFF_save_path = osp.join(SAVE_ROOT,"DFF")

    RCF_path= osp.join(EVAL_RES_ROOT,"precomputed/rcf")
    RCF_save_path = osp.join(SAVE_ROOT,"RCF")

    OFNET_path= osp.join(EVAL_RES_ROOT,"precomputed/ofnet")
    OFNET_save_path = osp.join(SAVE_ROOT,"OFNET")

    HED_path= osp.join(EVAL_RES_ROOT,"precomputed/hed")
    HED_save_path = osp.join(SAVE_ROOT,"HED")

    gt_save_path = osp.join(SAVE_ROOT,'GT')

 
    
    # maximun_for_contraint_loss(with_loss_path,with_loss_save_path)
    # maximun_for_contraint_loss(without_loss_path,without_loss_save_path)
    # concat_two_set(with_loss_save_path,without_loss_save_path)

    #* 绘制证明loss 有效的代码
    tmp = osp.join(SAVE_ROOT,"tmp")
    
    # path = osp.join(SAVE_ROOT,"loss_our_with_rind.json")
    # path = osp.join(SAVE_ROOT,"with_and_without_loss.json")
    # draw_topK_result(path,with_loss_path,with_loss_save_path,without_loss_path,without_loss_save_path,save_path=tmp)

    #! GT, OURS, RINDNET,DFF,RCF,*OFNet,HED 
    # draw_precision_according_best_performace(rindnet_path,TASKS,tmp)
    # best_dict = get_best_ois_for_each_task(edge_cerberus,TASKS)
    #* 绘制所有的, 

    
    paths = [edge_cerberus,RINDNET_path,DFF_path,RCF_path,OFNET_path,HED_path]
    save_paths = [edge_cerberus_save_path,RINDNET_save_path,DFF_save_path,RCF_save_path,OFNET_save_path,HED_save_path]
    # all_images =sorted( [x.split('.')[0] for x in os.listdir(osp.join(ORIGIN_IMG_GT,'depth'))])

    pick_imgs = ["376086","10081","112056","385022","179084"]
    
    # delete_dir(save_paths)
    # delete_dir([gt_save_path])
    
    for image_name in pick_imgs:
        for task in TASKS:
            draw_precision(paths,task,image_name,save_paths)        
    # draw_gt_map(all_images,gt_save_path)
 
            
    
    # save_paths = [edge_cerberus_save_path,RINDNET_save_path,DFF_save_path,RCF_save_path,OFNET_save_path,HED_save_path,gt_save_path]
    # tmp = osp.join(SAVE_ROOT,"concat_for_compare")
    # make_dir(tmp)
    # draw_compare_plot(all_images,save_paths,tmp)
    
    
            
            


                
        
        

    
    
    
        



    

    
    

    
    
    

    

    
    
 