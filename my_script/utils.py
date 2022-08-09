'''
Author: xushaocong
Date: 2022-08-04 16:42:24
LastEditTime: 2022-08-09 18:48:30
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




a= "/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/networks/edge_cerberus8/edge_without_constraint_losspth2_0"
# a= "/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/networks/edge_cerberus8/edge_final_3_3090pth2_0"

task=['reflectance','illumination','normal','depth','edge']


all_name = np.array(sorted([x.split('.')[0] for x in os.listdir(osp.join(a,'depth',"met"))]))


data = None
res = {}
all_dict = {}
for t in task:
    #* 写入的是  OIS 的[image,  threshold,Recall, Precision, F1, image_name]
    data = np.loadtxt(osp.join(a,t,"nms-eval","eval_bdry_img.txt"),dtype=np.str0)

    f1= data[:,-1].copy().astype(np.float64)
    f1_index = np.argsort(-f1)#* 降序排序的index
    
    # all_name = np.expand_dims(all_name,1)
    data = np.concatenate([data,all_name.reshape(len(all_name),1)],1)
    
    #* reshape 成一个字典
    data_dict = {x:y for x,y in zip(all_name,data)}

    
    for x in f1_index[:40]:
        data_dict[all_name[x]]
        res[all_name[x]] =  res.get(all_name[x],0)+1
    #* save dict 
    all_dict[t]=data_dict
    



t = sorted(res.items(),key = lambda x:x[1],reverse=True)
print(t[:10])
print(t[0][0])




# for k,v in all_dict.items():
#     print(v[t[0][0]])


for k,v in all_dict.items():
    print(v[t[0][0]])
    

        
    
# np.savetxt(osp.join(a,t,'img_ois.txt'),data,fmt='%s',delimiter='   ')
    




    