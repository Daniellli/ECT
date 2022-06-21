'''
Author: xushaocong
Date: 2022-06-16 17:33:26
LastEditTime: 2022-06-21 20:38:11
LastEditors: xushaocong
Description: 
FilePath: /cerberus/my_script/clear_trash.py
email: xushaocong@stu.xmu.edu.cn
'''

import os 
import os.path as osp 
from loguru import logger 

import shutil


'''
description:  删除target 目录下没有model的目录 
param {*} target
return {*}
'''
def clear_trash(target = "networks"):
    
    all_dir = [x for x in os.listdir(path=target)  if not osp.isfile(osp.join(target,x))]

    for p in all_dir:
        ckpg_dir = osp.join(target,p,"checkpoints")
        if  not osp.exists(ckpg_dir) or  6 > len(os.listdir(ckpg_dir)) :
            # logger.info(f"{ckpg_dir} empty")
            tmp = osp.join(target,p)
            shutil.rmtree(tmp, ignore_errors=False, onerror=None)
            logger.info(f"{tmp} has deleted  ")
        else :
            logger.info(f"{ckpg_dir} has models ")




    
if __name__ == "__main__":
    clear_trash()   
