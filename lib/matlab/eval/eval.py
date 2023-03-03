'''
Author: daniel
Date: 2023-02-27 22:26:35
LastEditTime: 2023-03-03 19:40:30
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




'''
description:  eval 
param {*} eval_dir
return {*}
'''
def eval_cityscapes(dirname='/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/cityscapes/cerberus/run_1',results_dir_name = None):
    eval_dir = [dirname]
    results_dir = []
    
    if results_dir_name is not None:
        results_dir.append(join(dirname,results_dir_name))
    else:     
        results_dir.append(join(dirname,'eval_res'))

    eng = matlab.engine.start_matlab()
    eval_res = eng.demoBatchEvalCityscapes(eval_dir,results_dir) #* 评估完会返回一串 string 

    with open(join(results_dir[0]+"res.json"),'w') as f :
        json.dump(eval_res,f)

    return eval_res


def eval_sbd(dirname='/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/sbd/cerberus/08:18:1677802680',results_dir_name = None):
    eval_dir = [dirname]
    results_dir = []
    
    if results_dir_name is not None:
        results_dir.append(join(dirname,results_dir_name))
    else:     
        results_dir.append(join(dirname,'eval_res'))

    eng = matlab.engine.start_matlab()
    eval_res = eng.demoBatchEvalSBD(eval_dir,results_dir) #* 评估完会返回一串 string 

    with open(join(results_dir[0]+"res.json"),'w') as f :
        json.dump(eval_res,f)

    return eval_res




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', default='cityscapes',choices=['sbd','cityscapes'])

    parser.add_argument('-d','--eval-dir',
                    default='/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/cityscapes/cerberus/run_1',
                    help="eval data dir, must be absolution dir ")

    

    args = parser.parse_args()

    tic = time.time()
    if args.dataset == 'sbd':
        aa = eval_sbd(args.eval_dir)
    elif args.dataset == 'cityscapes':
        aa = eval_cityscapes(args.eval_dir)
    logger.info(f'spend time : {time.strftime("%H:%M:%s",time.gmtime(time.time()-tic))}')
