'''
Author:   "  "
Date: 2022-07-21 19:21:58
LastEditTime: 2023-08-06 22:33:32
LastEditors: daniel
Description:  
FilePath: /Cerberus-main/plot/plot_precision_recall_plot.py
email:  
'''


import matlab
import matlab.engine
import argparse
import os.path as osp
import os
os.chdir("/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/plot")
from  loguru import logger

import shutil


'''
Description: Call MATLAB to perform evaluation.

Parameters:

eval_data_dir: Absolute path to the evaluation data directory.
Returns:

None
'''
def plot(test_edge = False):

    eng = matlab.engine.start_matlab()
    eng.plot_rind_edge() 
    if test_edge:
        eng.plot_rind_alledges()
    

'''
Description: Move the necessary files for drawing algorithm evaluation results to the required folder.

Parameters:

my_res: Folder containing the algorithm evaluation results.
prefix: Name of the algorithm result for drawing the curves.
Returns:

None
'''
def move_alg_res2plot_dir(
    my_res ="/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/networks/edge_cerberus_v8/model_res_2",
    prefix= "Ours",
    test_edge = False
    ):
    target= "/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/plot"
    
    suffix1 = "_bdry.txt"
    suffix2 = "_bdry_thr.txt"
    data = None
    if test_edge:
        data={
            "depth":osp.join(my_res,'depth',"nms-eval"),
            "illumination": osp.join(my_res,'illumination','nms-eval'),
            "normal":osp.join(my_res,'normal','nms-eval'),
            "reflectance":osp.join(my_res,'reflectance','nms-eval'),
            "rind_edges":osp.join(my_res,'edge','nms-eval')
        }
    else :
        data={
            "depth":osp.join(my_res,'depth',"nms-eval"),
            "illumination": osp.join(my_res,'illumination','nms-eval'),
            "normal":osp.join(my_res,'normal','nms-eval'),
            "reflectance":osp.join(my_res,'reflectance','nms-eval'),
        }

    
    #* move2teminal path 
    for k,v in data.items():
        target_dir  = osp.join(target,"eval_"+k)
        #* move source1,source2 to target_dir
        shutil.copyfile(osp.join(v,"eval_bdry.txt"),osp.join(target_dir,prefix+suffix1))
        shutil.copyfile(osp.join(v,"eval_bdry_thr.txt"),osp.join(target_dir,prefix+suffix2) )
        


if __name__ =="__main__":
    EVAL_RES_ROOT="/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/networks"

    # best_performance = osp.join(EVAL_RES_ROOT,"edge_cerberus8/edge_residualpth_0")
    edge_cerberus= osp.join(EVAL_RES_ROOT,"need2release/full_version_0")
    # edge_cerberus= '/home/DISCOVER_summer2022/xusc/exp/EDTER/work_dirs/EDTER_BIMLA_320x320_80k_bsds_rind_bs_8/10000'
    test_edge = False
    # move_alg_res2plot_dir(my_res=edge_cerberus,prefix='EDTER',test_edge = test_edge)
    # move_alg_res2plot_dir(my_res=edge_cerberus,test_edge = test_edge)

    plot(test_edge = test_edge)

   




    
    

    

    
     
    
    
    




    

    

    





