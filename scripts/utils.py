'''
Author:   "  "
Date: 2022-08-04 16:42:24
LastEditTime: 2023-08-06 22:37:02
LastEditors: daniel
Description: 
FilePath: /Cerberus-main/scripts/utils.py
email:  
'''



import json
import os
from re import S
from loguru import logger 

import os.path as osp

import numpy as np
import cv2

import json

import shutil
import glob

import math

import torch

from tqdm import tqdm

def get_arg_ois_score(all_dict,name,including_edge=True):
    tmp = []
    
    for idx,(k,v) in enumerate(all_dict.items()):
        if k == 'all_edges'  and not including_edge:
            break
        tmp.append(float(v[name][1]))
    
    return np.array(tmp).mean()
    



def test():
    data= {
        "depth": {
            "ODS": "0.658",
            "OIS": "0.698",
            "AP": "0.640",
        },
        "normal": {
            "ODS": "0.454",
            "OIS": "0.514",
            "AP": "0.387",
        },
        "reflectance": {
            "ODS": "0.437",
            "OIS": "0.492",
            "AP": "0.350"
        },
        "illumination": {
            "ODS": "0.226",
            "OIS": "0.315",
            "AP": "0.142"
        }
    }


    
    ods = ois= ap=0
    for k,v in data.items():
        ods+= float(v["ODS"])
        ois+=float(v["OIS"])
        ap+=float(v["AP"])


    data["Average"] = {} 
    data["Average"]["ODS"] = round( ods/4,3)
    data["Average"]["OIS"] = round(ois/4,3)
    data["Average"]["AP"] = round(ap/4,3)

    logger.info(data['Average'])



'''
Description: Get the evaluation results.

Parameters:

path: Path where the results are stored.
tasks: List of tasks to evaluate.
print_top10: Whether to print the top 10 results.
avg_including_edge: Whether to include edges when evaluating precision.
Returns:

None
'''
def get_eval_res(path, 
                tasks=['reflectance','illumination','normal','depth'],
                print_top10=False,
                avg_including_edge=False
                ):
    
    
    all_name = np.array(sorted([x.split('.')[0] for x in os.listdir(osp.join(path,'depth',"nms"))]))

    data = None
    all_dict = {}
    
    for t in tasks:
        
        data = np.loadtxt(osp.join(path,t,"nms-eval","eval_bdry_img.txt"),dtype=np.str0)

        
        # f1= data[:,-1].copy().astype(np.float64)
        # f1_index = np.argsort(-f1)#* 降序排序的index
        
        
        data_dict = {x:y for x,y in zip(all_name,data)}

        #* save dict 
        all_dict[t]=data_dict
        

    
    avg_ ={}
    for name in all_name:
        avg_[name] = round(get_arg_ois_score(all_dict,name,including_edge=avg_including_edge),3,)

    
    if print_top10:
    
        idx = np.argsort(-np.array(list(avg_.values())))
        for a in idx[:10]:
            name = list(avg_.keys())[a]
            print(name,avg_[name])

    
            for  k,v in all_dict.items():
                print(v[name])
    return  all_dict,avg_
            



def print_topX_avg_ois(X,avg_ois,eval_dict,print_all=False):
    best_avg_ois_idx = np.argsort(-np.array(list(avg_ois.values())))
    for idx in best_avg_ois_idx[:X]:
        name = list(avg_ois.keys())[idx]
        print(f"name :{name } \t   avg performance : {avg_ois[name]}")
        
        if print_all:
            for k,v in eval_dict.items():
                print(v[name])
            print("==============================================================")



def search_dict(_dict,search_name,return_numpy = True):
    res = {}
    for k,v in _dict.items():
        res[k]= v[search_name].tolist()
    
    if return_numpy:
        res = {k:  np.array((v[0],v[1],v[2],v[3],v[4]),
                dtype=[('img_idx',np.float32),('OIS',np.float64),('recall',np.float64),('precision',np.float64),('F1',np.float64)]) 
                for k,v in res.items()}
    else:
        pass

    
    
    return res
        

def dump_dict(json_data,filename):
    
    with open(filename,'w') as f :
        json.dump(json_data,f)
    
    
'''
description:  Given two sets of data, the task is to output the top K pairs with the maximum distance.
 The distance is calculated by subtracting the values from group B from group A.
  In other words, we are looking for the pairs where A outperforms B the most.
param {*} K
param {*} A_avg_ois
param {*} A_eval_dict
param {*} B_avg_ois
param {*} B_eval_dict
return {*}
'''
def print_topK_distance(K,A_avg_ois,A_eval_dict,B_avg_ois,B_eval_dict,save_path =None):
    minus_res = {k: round(v - B_avg_ois[k],3) for k,v in A_avg_ois.items()}
    distance_sorted_idx=np.argsort(-np.array(list(minus_res.values())))
    res = {}
    for  name_idx in  distance_sorted_idx[:K]:
        name = list(minus_res.keys())[name_idx]
        A_avg_ois_score = round(get_arg_ois_score(A_eval_dict,name),3)
        B_avg_ois_score = round(get_arg_ois_score(B_eval_dict,name),3)
        print(f" name : {name} \t group A  score : {A_avg_ois_score} \t group B  score: {B_avg_ois_score}, \t distance = {minus_res[name]}")
        pair = {}
        pair['A1'] = search_dict(A_eval_dict,name,return_numpy=False)
        pair['B1'] = search_dict(B_eval_dict,name,return_numpy=False)
        res[name] = pair

    if save_path is not None:
        dump_dict(res,osp.join(SAVE_ROOT,save_path))

    return res



'''
description:  Given an evaluation dataset, the task is to obtain the image with the best OIS (Overall Image Quality Score) for each task.
param {*} eval_dict
return {*}
'''
def __get_best_ois_for_each_task(eval_dict,top_k=1):

    res = {}
    for task,all_evals in eval_dict.items():
        #* get  the best ios  in single task 
        all_ois_data = {}
        for img_name,eval_data in all_evals.items():
            all_ois_data[img_name]= float(eval_data[1])
        #* sort
        all_ois_data = sorted(all_ois_data.items(),key=lambda k : k[1],reverse=True)
        logger.info(f" task :{task}, img name : {all_ois_data[0][0]}, ois: {all_ois_data[0][1]}")
        res[task]=all_ois_data[:top_k]
    return res
        
        
    

def __get_ois_threshold_accoding_name(image_name,path,task): 
    eval_dict,_=get_eval_res(path,[task])#* single task 
    return float(eval_dict[task][image_name][1])
    
    


def get_ois_threshold_accoding_name(image_name,path,task):
    return __get_ois_threshold_accoding_name(image_name,path,task)


def get_best_ois_for_each_task(path,task):
    eval_dict,_=get_eval_res(path,task)
    return __get_best_ois_for_each_task(eval_dict)
    
    


def __get_ods(path):

    with open(osp.join(path,'eval_res.json'),'r')as f :
        data = json.load(f)
    
    return  data



def __get_task_ods(path,task):
    ods  = __get_ods(path)

    
    return float(ods[task]['ODS'])

    

def get_task_ods(path,task):
    return __get_task_ods(path,task)


def print_topK_distance_in_specific_task(pathA,pathB,task,K=10):
    
    dictA ,_ = get_eval_res(pathA,[task])
    dictB ,_ = get_eval_res(pathB,[task])

    A = { name:float(value[1]) for name,value in dictA[task].items()}
    B = { name:float(value[1]) for name,value in dictB[task].items()}


    minus_res = {k: round(v - B[k],3) for k,v in A.items()}
    distance_sorted_idx=np.argsort(-np.array(list(minus_res.values())))
    
    
    for  name_idx in  distance_sorted_idx[:K]:
        name = list(minus_res.keys())[name_idx]
        logger.info(f"task:{task} \t  name : {name} \t distance = {minus_res[name]}")
        
        



def print_topK_distance_in_specific_task_multi_source(pathA , paths,task,K=10):
    

    dictA ,_ = get_eval_res(pathA,[task])
    A = { name:float(value[1]) for name,value in dictA[task].items()}
    
    dicts = []
    for p in paths:
        tmp ,_ = get_eval_res(p,[task])
        dicts.append({name:float(value[1]) for name,value in tmp[task].items()})
        
   
   #* A compare to all in this task
    distances={}
    mean_distance = {}
    for img_name,osi in A.items():
        dis = []
        for d in dicts:
            dis.append(osi - d[img_name])
        distances[img_name] = dis
        mean_distance[img_name] = np.array(dis).mean()
        # logger.info(distances)
        # logger.info(mean_distance[img_name] )
        
    
    
    #* sort according to the mean distance 
    distance_sorted_idx=np.argsort(-np.array(list(mean_distance.values())))
    
    #*  print top K 
    for  name_idx in  distance_sorted_idx[:K]:
        name = list(mean_distance.keys())[name_idx]
        logger.info(f"task:{task} \t  name : {name} \t distance = {mean_distance[name]}")
        logger.info(distances[name])


    


def interp_img(img,to_size= (320,480)):
    a = torch.from_numpy(img)
    a = a.permute([2,0,1]).unsqueeze(0)
    a = torch.nn.functional.interpolate(a,size=to_size)
    a = a.squeeze().permute(1,2,0).numpy()
    return a 

'''
Description: Convert a video into a sequence of images.

Parameters:

video_path: Path of the video file.
im_dir: Directory to store the converted images.
img_name_format: Format of the file name for the converted images, e.g., "dancer-1.jpg".
interp: Whether to perform downsampling. If the images are too large, downsampling can be applied.
Returns:

None
'''
def video2imgs(video_path , im_dir,img_name_format="%06d.png" ,interp=False):
    
    if ( not osp.exists(im_dir)):
        os.mkdir(im_dir)
        
        
    cap = cv2.VideoCapture(video_path)
    ind =  1
    ret, frame = cap.read()
    while(ret):
        if interp:
            a  =interp_img(frame)
        else :
            a =frame
            
        cv2.imwrite(osp.join(im_dir,img_name_format%ind), a)
        ind+=1
        ret, frame = cap.read()
    cap.release()
    cv2.destroyAllWindows()
    return ind# 产生了多少帧图像





def make_dir(path):
    if  not osp.exists(path):
        os.makedirs(path)



def delete_dirs_files(asb_path_list):
    
    for d in need_delete_p:
        if osp.isdir(d):
            shutil.rmtree(d)
        else:
            os.remove(d)
        print(d)



def inter_dir(dir_path):
    target_p = osp.join(osp.dirname(dir_path),'after_interp')
    make_dir(target_p)
    all_imgs = sorted(glob.glob(dir_path+"/*.jpg"))
    for idx,im in tqdm(enumerate(all_imgs)):
        print(im)
        a = interp_img(cv2.imread(im))
        cv2.imwrite(osp.join(target_p,"%06d.png"%(idx)),a)
        
        
    
    



def my_video2img(dir_path="/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/plot/demo"):
    all_mp4 = glob.glob(dir_path+"/*.mp4")
    for  idx,p in tqdm(enumerate(all_mp4)):
        logger.info(f"processing  {p}")
        save_p = osp.join(osp.dirname(p),'%04d'%(idx),'imgs')
        
        make_dir(save_p)
        video2imgs(p,save_p,interp=True)




if __name__ == "__main__":


    SAVE_ROOT = "/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/plot-rind-edge-pr-curves/plot_material"
    EVAL_RES_ROOT="/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/networks"
    TASKS = ["reflectance","illumination","normal","depth","all_edges"]

    RINDNET_path= osp.join(EVAL_RES_ROOT,"precomputed/rindnet-resnet50")
    DFF_path= osp.join(EVAL_RES_ROOT,"precomputed/dff")
    RCF_path= osp.join(EVAL_RES_ROOT,"precomputed/rcf")
    OFNET_path= osp.join(EVAL_RES_ROOT,"precomputed/ofnet")
    HED_path= osp.join(EVAL_RES_ROOT,"precomputed/hed")


    # edge_cerberus= osp.join(EVAL_RES_ROOT,"edge_cerberus8/edge_final_3_3090pth_0")
    edge_cerberus= osp.join(EVAL_RES_ROOT,"final_version/edge_final_8_3090_0")
    without_constraint_loss = osp.join(EVAL_RES_ROOT,"final_version/edge_final_4_A100_80G_no_loss_0")

    #* 子任务
    # our_eval_dict,our_avg_ois=get_eval_res(edge_cerberus,TASKS,avg_including_edge=False)
    # no_loss_eval_dict,no_loss_avg_ois=get_eval_res(without_constraint_loss,TASKS,avg_including_edge=False)
    # rindnet_eval_dict,rindnet_avg_ois=get_eval_res(RINDNET_path,TASKS,avg_including_edge=False)
    # print_topX_avg_ois(5,our_avg_ois,our_eval_dict,print_all=True)

    # # print_topX_avg_ois(5,no_loss_avg_ois,no_loss_eval_dict)
    # print_topK_distance(5,our_avg_ois,our_eval_dict,rindnet_avg_ois,rindnet_eval_dict)


    # print_topK_distance(10,our_avg_ois,our_eval_dict,
    #                 rindnet_avg_ois,rindnet_eval_dict,
    #                 save_path = "loss_our_with_rind.json") 


    
    # __get_best_ois_for_each_task(our_eval_dict)
    # get_ois_threshold_accoding_name('2018',ours_res_path,'depth')
    
    
    # paths = [RINDNET_path,DFF_path,RCF_path,OFNET_path,HED_path]
    # for t in TASKS[4:5]:
        # print_topK_distance_in_specific_task(edge_cerberus,RINDNET_path,t,K=20)
        # print_topK_distance_in_specific_task_multi_source(edge_cerberus,paths,t,K=10)


    # p = "/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/plot/demo/demo1/Transform_video_Q15-Img"
    # p = "/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/plot/demo/demo%d"
    # p = "/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/plot/demo/%04d"
    # for i in range(0,5):
    #     need_delete_p = glob.glob(osp.join(p%i,'2022*'))
    #     delete_dirs_files(need_delete_p)


    

    # my_video2img()

    

    path = "/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/plot/demo/KITTI/KITTI_21_OR_22"
    save_path = osp.join(osp.dirname(path),'imgs')
    make_dir(save_path)
    all_img= sorted(glob.glob(path+"/*.png"))
    for idx,im in tqdm(enumerate(all_img)):
        tmp  = cv2.imread(im)

        H,W,C = tmp.shape

        to_size = (math.ceil(H/32)*32,math.ceil(W/32)*32)
        
        
        tmp2 = interp_img(tmp,to_size)
        cv2.imwrite(osp.join(save_path,"%06d.png"%idx),tmp2)
        if idx == 0 :
            logger.info(f"from {H,W} to {to_size }")
        






