'''
Author:   "  "
Date: 2022-06-30 23:04:50
LastEditTime: 2022-06-30 23:26:31
LastEditors:   "  "
Description: 
FilePath: /cerberus/utils/check_model_consistent.py
email:  
'''


#!/usr/bin/env python
# -*- coding: utf-8 -*-


import torch


'''
description:  判断两个model的参数是否一致, 
param {*} checkpoint_0_path
param {*} checkpoint_1_path
return {*} 返回 三个参数, 1: 是否一致, 2. 不一致的参数个数, 3. 不一致的参数key
'''
def is_model_consistent(checkpoint_0_path,checkpoint_1_path):

    checkpoint_0 = torch.load(checkpoint_0_path)
    checkpoint_1 = torch.load(checkpoint_1_path)
    cnt = []
    keys = []
    
    for key in checkpoint_0['state_dict'].keys():
        
        r0 = checkpoint_0['state_dict'][key].cpu().numpy()
        r1 = checkpoint_1['state_dict'][key].cpu().numpy()
        tmp = (r0!=r1).sum()
        if tmp!= 0 :
            
            keys.append(key)
            cnt.append(tmp)

    return len(keys)==0, cnt,keys
         