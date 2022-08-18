'''
Author: xushaocong
Date: 2022-08-04 16:42:24
LastEditTime: 2022-08-17 10:08:04
LastEditors: xushaocong
Description: 
FilePath: /Cerberus-main/my_script/utils.py
email: xushaocong@stu.xmu.edu.cn
'''



import json
import os
from loguru import logger 

import os.path as osp

import numpy as np



def get_arg_ois_score(all_dict,name):
    tmp = []
    for k,v in all_dict.items():
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
description: 获取评估的结果 
param {*} path : 结果存储的路径
param {*} tasks: list  : 要评估的任务
param {*} print_top10 : 是否打印前10个
return {*}
'''
def get_eval_res(path, 
                tasks=['reflectance','illumination','normal','depth'],
                print_top10=False):
    
    #* 获取所有图像的名字
    all_name = np.array(sorted([x.split('.')[0] for x in os.listdir(osp.join(path,'depth',"nms"))]))

    data = None
    all_dict = {}
    #* 将所有任务所有图片评估的iso 整合成一个dict
    for t in tasks:
        #* 写入的是  OIS 的[image,  threshold,Recall, Precision, F1, image_name]
        data = np.loadtxt(osp.join(path,t,"nms-eval","eval_bdry_img.txt"),dtype=np.str0)

        #* f1 score , 每张图在这个任务上的ois score  进行排序, 
        # f1= data[:,-1].copy().astype(np.float64)
        # f1_index = np.argsort(-f1)#* 降序排序的index
        
        #* reshape 成一个字典
        data_dict = {x:y for x,y in zip(all_name,data)}

        #* save dict 
        all_dict[t]=data_dict
        

    #* 获取每张图像 5个子任务的 ois score 的avg score 
    avg_ ={}
    for name in all_name:
        avg_[name] = round(get_arg_ois_score(all_dict,name),3)

    #*  输出前10个 
    if print_top10:
        #* 对avg score 排序
        idx = np.argsort(-np.array(list(avg_.values())))
        for a in idx[:10]:
            name = list(avg_.keys())[a]
            print(name,avg_[name])

            #* 输出单个的数值
            for  k,v in all_dict.items():
                print(v[name])
    return  all_dict,avg_
            


# a= "/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/networks/edge_cerberus8/edge_without_constraint_losspth2_0"
ours_res_path= "/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/networks/edge_cerberus8/edge_final_3_3090pth_0"
#* 算法结果
rindnet_res_path="/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/networks/precomputed/rindnet-resnet50"

#* 子任务
# task=['reflectance','illumination','normal','depth','edge']

our_eval_dict,our_avg_ois=get_eval_res(ours_res_path)
rindnet_eval_dict,rindnet_avg_ois=get_eval_res(rindnet_res_path)


minus_res = {k: round(v - rindnet_avg_ois[k],3) for k,v in our_avg_ois.items()}

distance_sorted_idx=np.argsort(-np.array(list(minus_res.values())))
print("hello")



for  name_idx in  distance_sorted_idx[:5]:
    name = list(minus_res.keys())[name_idx]
    our_avg_ois_score = round(get_arg_ois_score(our_eval_dict,name),3)
    rindnet_avg_ois_score = round(get_arg_ois_score(rindnet_eval_dict,name),3)
    
    print(f" name : {name} \t our score : {our_avg_ois_score} \t rindnet score: {rindnet_avg_ois_score}, \tdistance = {minus_res[name]}")
    




# need_to_knowns = {'185092':0.865 , '141048': 0.862,"250047":0.859,"238025":0.856,"326085":0.855}#* need to known in rindnet 
#todo : 查看185092, ours: 185092 0.865; 141048 0.862;250047 0.859;238025 0.856;326085 0.855
#* 差距比较大的,   185092,238025,326085,250047
    

