'''
Author:   "  "
Date: 2022-05-07 15:08:33
LastEditTime: 2022-06-13 20:17:42
LastEditors:   "  "
Description: 
FilePath: /Cerberus-main/my_script/sweep_utils.py
email:  
'''
# from my_script.sweep_test import run_from_train
import traceback
import time
import torch
from loguru import logger 
import wandb

import os

params_map = {
    "seed": "sd",
    "lr": "lr",
    "batch_size": "bs",
    "epochs": "ep",
}


'''
description:  初始化代理
return {*}
'''
def init_agent():
    run = wandb.init(job_type="Training")
    config = wandb.config
    run.name = run.project
    #* 给run.name 换一个唯一的名字, 以参数命名
    for k, v in config.__dict__["_items"].items():
        if "wandb" not in k:
            if k in params_map.keys():
                run.name += "_%s@%s" % (str(params_map[k]), str(v))            
    return run, config

'''
description: 
param {*} log_file
param {*} message
return {*}
'''
def log_info(log_file,message):
    logger.info(message)




def clear_expr_cache(run, e=None):
    del run
    if e is not None:
        traceback.print_exc()
        logger.info(e)
        torch.cuda.empty_cache()
        time.sleep(60)
