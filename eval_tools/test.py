'''
Author: xushaocong
Date: 2022-06-13 10:30:59
LastEditTime: 2022-07-18 11:21:23
LastEditors: xushaocong
Description:  使用matlab engin 进行eval
FilePath: /Cerberus-main/eval_tools/test.py
email: xushaocong@stu.xmu.edu.cn
'''




import matlab
import matlab.engine
import argparse
import os 
import os.path as osp
import sys
import json 
from  loguru import logger
parser = argparse.ArgumentParser(description='')
parser.add_argument('-d', '--eval-data-dir', 
    default='./dataset/BSDS_RIND_mine',help="eval data dir  , must be absolution dir ")
args = parser.parse_args()

# logger.info(osp.dirname(__file__))
# sys.path.append(osp.dirname(__file__)) #* 加这行无效!
# sys.path.append(osp.join(osp.dirname(__file__),"edges")) #* 加这行无效!
# sys.path.append(osp.join(osp.dirname(__file__),"edges","private2")) #* 加这行无效!
# os.chdir(osp.dirname(__file__)) #* 需要到当前目录才能执行??? 


'''
description:  调用matlab 来eval , 
param {*} eval_data_dir : 只能绝对路径进行测试, 
return {*}
'''
def test_by_matlab(eval_data_dir):
    logger.info(eval_data_dir)
    eng = matlab.engine.start_matlab()
    # keys=['depth','normal','reflectance','illumination']
    keys=['depth','normal','reflectance','illumination','edge']
    eval_res = eng.eval_edge(eval_data_dir,keys) #* 评估完会返回一串 string 
    res = {}
    sum_ODS = sum_OIS = sum_AP =sum_R50 = 0
    
    for idx, eval_value in enumerate(eval_res):#* ODS, OIS, AP, R50        
        res[keys[idx]] = {"ODS": "%.3f"%(eval_value[0]),"OIS":  "%.3f"%(eval_value[1]),
            "AP": "%.3f"%(eval_value[2]),"R50":"%.3f"%(eval_value[3])}
        
        if idx !=4:
            sum_ODS+= eval_value[0]
            sum_OIS+= eval_value[1]
            sum_AP+= eval_value[2]
            sum_R50+= eval_value[3] if  eval_value[3]  is not None else 0 

    num_sub_task = 4
    res["Average"]= {
        "ODS": sum_ODS/num_sub_task,
        "OIS":sum_OIS/num_sub_task,
        "AP":sum_AP/num_sub_task,
        "R50":sum_R50/num_sub_task 
    }
    #* calc average 

    
    
    with open (osp.join(eval_data_dir,"eval_res.json"),'w')as f :
        json.dump(res,f)
    return res


if __name__ =="__main__":
    test_by_matlab(args.eval_data_dir)
    # test_by_matlab("/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/networks/model_res")
    # test_by_matlab("/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/networks/dashing-wind-713/model_res")
    

    





