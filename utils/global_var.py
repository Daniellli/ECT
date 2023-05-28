'''
Author:   "  "
Date: 2022-06-20 22:41:22
LastEditTime: 2022-06-20 22:44:46
LastEditors:   "  "
Description: 
FilePath: /Cerberus-main/utils/global_var.py
email:  
'''
import numpy as np


ddp_file = "ddp.json"

#*====================
TASK =None  # 'ATTRIBUTE', 'AFFORDANCE', 'SEGMENTATION' 
TRANSFER_FROM_TASK = None  #'ATTRIBUTE', 'AFFORDANCE', 'SEGMENTATION', or None to unable transfer

CITYSCAPE_PALETTE = np.asarray([
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [70, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32],
    [0, 0, 0]], dtype=np.uint8)

NYU40_PALETTE = np.asarray([
    [0, 0, 0], 
    [0, 0, 80], 
    [0, 0, 160], 
    [0, 0, 240], 
    [0, 80, 0], 
    [0, 80, 80], 
    [0, 80, 160], 
    [0, 80, 240], 
    [0, 160, 0], 
    [0, 160, 80], 
    [0, 160, 160], 
    [0, 160, 240], 
    [0, 240, 0], 
    [0, 240, 80], 
    [0, 240, 160], 
    [0, 240, 240], 
    [80, 0, 0], 
    [80, 0, 80], 
    [80, 0, 160], 
    [80, 0, 240], 
    [80, 80, 0], 
    [80, 80, 80], 
    [80, 80, 160], 
    [80, 80, 240], 
    [80, 160, 0], 
    [80, 160, 80], 
    [80, 160, 160], 
    [80, 160, 240], [80, 240, 0], [80, 240, 80], [80, 240, 160], [80, 240, 240], 
    [160, 0, 0], [160, 0, 80], [160, 0, 160], [160, 0, 240], [160, 80, 0], 
    [160, 80, 80], [160, 80, 160], [160, 80, 240]], dtype=np.uint8)

AFFORDANCE_PALETTE = np.asarray([
    [0, 0, 0],
    [255, 255, 255]], dtype=np.uint8)




task_list = None
middle_task_list = None

if TASK == 'ATTRIBUTE':
    task_list = ['Wood','Painted','Paper','Glass','Brick','Metal','Flat','Plastic','Textured','Glossy','Shiny']
    FILE_DESCRIPTION = '_attribute'
    PALETTE = AFFORDANCE_PALETTE
    EVAL_METHOD = 'mIoU'
elif TASK == 'AFFORDANCE':
    task_list = ['L','M','R','S','W']
    FILE_DESCRIPTION = '_affordance'
    PALETTE = AFFORDANCE_PALETTE
    EVAL_METHOD = 'mIoU'
elif TASK =='SEGMENTATION':
    task_list = ['Segmentation']
    FILE_DESCRIPTION = ''
    PALETTE = NYU40_PALETTE
    EVAL_METHOD = 'mIoUAll'
else:
    task_list = None
    FILE_DESCRIPTION = ''
    PALETTE = None
    EVAL_METHOD = None

if TRANSFER_FROM_TASK == 'ATTRIBUTE':
    middle_task_list = ['Wood','Painted','Paper','Glass','Brick','Metal','Flat','Plastic','Textured','Glossy','Shiny']
elif TRANSFER_FROM_TASK == 'AFFORDANCE':
    middle_task_list = ['L','M','R','S','W']
elif TRANSFER_FROM_TASK =='SEGMENTATION':
    middle_task_list = ['Segmentation']
elif TRANSFER_FROM_TASK is None:
    pass


# if TRANSFER_FROM_TASK is not None:
#     TENSORBOARD_WRITER = SummaryWriter(comment='From_'+TRANSFER_FROM_TASK+'_TO_'+TASK)
# elif TASK is not None:
#     TENSORBOARD_WRITER = SummaryWriter(comment=TASK)
# else:
#     TENSORBOARD_WRITER = SummaryWriter(comment='Nontype')

