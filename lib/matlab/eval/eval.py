'''
Author: daniel
Date: 2023-02-27 22:26:35
LastEditTime: 2023-02-27 22:56:11
LastEditors: daniel
Description: 
FilePath: /Cerberus-main/lib/matlab/eval/eval.py
have a nice day
'''




import matlab
import matlab.engine
import argparse
import os 
from  os.path import join,split,isdir,isfile


import sys
import json 
from  loguru import logger


import time
from IPython import embed 






# parser = argparse.ArgumentParser(description='')
# parser.add_argument('-d','--eval-data-dir',default='./dataset/BSDS_RIND_mine',help="eval data dir, must be absolution dir ")
# parser.add_argument('-d','--eval-data-dir',default='./dataset/BSDS_RIND_mine',help="eval data dir, must be absolution dir ")
# args = parser.parse_args()






'''
description:  eval 
param {*} eval_dir
return {*}
'''
def eval_cityscapes(eval_dir=['/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/cityscapes/cerberus/run_1'],results_dir_name = None):
    results_dir = []
    for dirname in eval_dir:
        if results_dir_name is not None:
            results_dir.append(join(dirname,results_dir_name))
        else:     
            results_dir.append(join(dirname,'eval_res'))

    eng = matlab.engine.start_matlab()
    eval_res = eng.demoBatchEvalCityscapes(eval_dir,results_dir) #* 评估完会返回一串 string 

    with open(join(results_dir[0]+"res.json"),'w') as f :
        json.dump(eval_res,f)

    return eval_res




if __name__ == "__main__":

    tic = time.time()
    aa = eval_cityscapes()
    logger.info(f'spend time : {time.strftime("%H:%M:%s",time.gmtime(time.time()-tic))}')
