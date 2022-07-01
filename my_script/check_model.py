'''
Author: xushaocong
Date: 2022-06-30 23:22:08
LastEditTime: 2022-06-30 23:27:15
LastEditors: xushaocong
Description: 

FilePath: /cerberus/my_script/check_model.py
email: xushaocong@stu.xmu.edu.cn
'''



import os
import os.path as osp
import sys
# os.chdir("/DATA2/xusc/exp/cerberus")
sys.path.append("/DATA2/xusc/exp/cerberus")


from utils.check_model_consistent import is_model_consistent
from loguru import logger


p = "/DATA2/xusc/exp/cerberus/networks/lr@1e-05_ep@5_bgw@1_rindw@1_1656601417/checkpoints"
p = [ osp.join(p,x) for x in os.listdir(p) if x.startswith("ckpt")]

logger.info(is_model_consistent(*p))

