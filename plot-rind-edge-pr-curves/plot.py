'''
Author: xushaocong
Date: 2022-07-21 19:21:58
LastEditTime: 2022-07-21 21:59:52
LastEditors: xushaocong
Description: 
FilePath: /Cerberus-main/plot-rind-edge-pr-curves/plot.py
email: xushaocong@stu.xmu.edu.cn
'''


import matlab
import matlab.engine
import argparse
import os.path as osp
from  loguru import logger

import shutil

'''
description:  调用matlab 来eval , 
param {*} eval_data_dir : 只能绝对路径进行测试, 
return {*}
'''
def plot():

    eng = matlab.engine.start_matlab()
    eng.plot_rind_edge() 
    eng.plot_rind_alledges()
    

'''
description:  将算法评估结果的绘图需要的文件移动到绘图需要的文件夹
my_res: 就是算法评估的结果文件夹
prefix: 绘制曲线图时我们的算法结果名字
return {*}
'''
def move_alg_res2plot_dir(
    my_res ="/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/networks/edge_cerberus_v8/model_res_2",
    prefix= "EdgeCerberus"
    ):
    target= "/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/plot-rind-edge-pr-curves"
    
    suffix1 = "_bdry.txt"
    suffix2 = "_bdry_thr.txt"

    data={
        "depth":osp.join(my_res,'depth',"nms-eval"),
        "illumination": osp.join(my_res,'illumination','nms-eval'),
        "normal":osp.join(my_res,'normal','nms-eval'),
        "reflectance":osp.join(my_res,'reflectance','nms-eval'),
        "rind_edges":osp.join(my_res,'edge','nms-eval')
    }
    
    #* move2teminal path 
    for k,v in data.items():
        target_dir  = osp.join(target,"eval_"+k)
        #* move source1,source2 to target_dir
        shutil.copyfile(osp.join(v,"eval_bdry.txt"),osp.join(target_dir,prefix+suffix1))
        shutil.copyfile(osp.join(v,"eval_bdry_thr.txt"),osp.join(target_dir,prefix+suffix2) )
        


if __name__ =="__main__":
    # move_alg_res2plot_dir()
    plot()

   




    
    

    

    
     
    
    
    




    

    

    





