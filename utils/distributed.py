'''
Author: xushaocong
Date: 2022-06-12 14:47:04
LastEditTime: 2022-06-12 16:13:50
LastEditors: xushaocong
Description:  分布式训练
FilePath: /Cerberus-main/utils/distributed.py
email: xushaocong@stu.xmu.edu.cn
'''
# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
from loguru import logger
import socket
import logging
import torch
import torch.nn as nn
import torch.distributed as dist


logger = logging.getLogger('global')


'''
description: 
param {*} v
return {*}
'''
def average_reduce(v):
    if get_world_size() == 1:
        return v
    tensor = torch.cuda.FloatTensor(1)
    tensor[0] = v
    dist.all_reduce(tensor)
    v = tensor[0] / get_world_size()
    return v


'''
description:  封装model
return {*}
'''
class DistModule(nn.Module):
    def __init__(self, module, bn_method=0):
        super(DistModule, self).__init__()
        self.module = module
        self.bn_method = bn_method
        if get_world_size() > 1:
            broadcast_params(self.module)
        else:
            self.bn_method = 0  # single proccess

    def forward(self, *args, **kwargs):
        broadcast_buffers(self.module, self.bn_method)
        return self.module(*args, **kwargs)

    def train(self, mode=True):
        super(DistModule, self).train(mode)
        self.module.train(mode)
        return self


def broadcast_params(model):
    """ broadcast model parameters """
    for p in model.state_dict().values():
        dist.broadcast(p, 0)


def broadcast_buffers(model, method=0):
    """ broadcast model buffers """
    if method == 0:
        return

    world_size = get_world_size()

    for b in model._all_buffers():
        if method == 1:  # broadcast from main proccess
            dist.broadcast(b, 0)
        elif method == 2:  # average
            dist.all_reduce(b)
            b /= world_size
        else:
            raise Exception('Invalid buffer broadcast code {}'.format(method))


inited = False

'''
description:  初始化参数 , 
param {*} debug
return {*}
'''
def _dist_init(debug=False):
    '''
    if guess right:
        ntasks: world_size (process num)
        proc_id: rank
    '''
    if debug :
        return 0, 1
    else :
        rank = int(os.environ['RANK'])
        num_gpus = torch.cuda.device_count()
        torch.cuda.set_device(rank % num_gpus)
        dist.init_process_group(backend='nccl')
        world_size = dist.get_world_size()
        print(f"num_gpus = {num_gpus} ,world_size={world_size},rank={rank}")
        return rank, world_size
    

def _get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip


# def dist_init():
def dist_init(debug=False):
    global rank, world_size, inited
    try:
        # rank, world_size = _dist_init()
        rank, world_size = _dist_init(debug)
        #!=================
    except RuntimeError as e:
        if 'public' in e.args[0]:
            logger.info(e)
            logger.info('Warning: use single process')
            rank, world_size = 0, 1
        else:
            raise RuntimeError(*e.args)
    #!==============
    inited = True
    return rank, world_size
    #!==============


def get_rank():
    if not inited:
        raise(Exception('dist not inited'))
    return rank


def get_world_size():
    if not inited:
        raise(Exception('dist not inited'))
    return world_size


def reduce_gradients(model, _type='sum'):
    types = ['sum', 'avg']
    assert _type in types, 'gradients method must be in "{}"'.format(types)
    logger("gradients method is {}".format(_type))
    if get_world_size() > 1:
        for param in model.parameters():
            if param.requires_grad:
                dist.all_reduce(param.grad.data)
                if _type == 'avg':
                    param.grad.data /= get_world_size()
    else:
        return None
