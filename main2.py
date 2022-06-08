'''
Author: xushaocong
Date: 2022-06-07 17:03:15
LastEditTime: 2022-06-08 08:01:40
LastEditors: xushaocong
Description:  dataloader 改成rindnet的
FilePath: /Cerberus-main/main2.py
email: xushaocong@stu.xmu.edu.cn
'''
#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import OrderedDict
import argparse
import json
import logging
import math
import os

# import pdb
from os.path import exists, join, split
import threading
from datetime import datetime

import time

import numpy as np
import shutil
import wandb
import sys
from PIL import Image
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.nn.modules import transformer
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from min_norm_solvers import MinNormSolver

import drn
import data_transforms as transforms
from model.models import DPTSegmentationModel, DPTSegmentationModelMultiHead, TransferNet, CerberusSegmentationModelMultiHead
from model.transforms import PrepareForNet

import os
import os.path as osp

# from torchcam.methods import SmoothGradCAMpp
import torch.nn.functional as F
try:
    from modules import batchnormsync
except ImportError:
    pass



#*====================

from dataloaders.datasets.bsds_hd5 import Mydataset


from torch.utils.data.distributed import DistributedSampler
run = wandb.init(project="train_cerberus") 
cur_dir=osp.dirname(__file__)
print(f"cur_dir = {cur_dir}")
#*====================
FORMAT = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
logging.basicConfig(format=FORMAT, filename=osp.join(cur_dir,'logs', run.name +"_"+ datetime.now().strftime("%Y%m%d_%H%M%S") + '.txt'))

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

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


if TRANSFER_FROM_TASK is not None:
    TENSORBOARD_WRITER = SummaryWriter(comment='From_'+TRANSFER_FROM_TASK+'_TO_'+TASK)
elif TASK is not None:
    TENSORBOARD_WRITER = SummaryWriter(comment=TASK)
else:
    TENSORBOARD_WRITER = SummaryWriter(comment='Nontype')

def downsampling(x, size=None, scale=None, mode='nearest'):
    if size is None:
        size = (int(scale * x.size(2)) , int(scale * x.size(3)))
    h = torch.arange(0,size[0]) / (size[0] - 1) * 2 - 1
    w = torch.arange(0,size[1]) / (size[1] - 1) * 2 - 1
    grid = torch.zeros(size[0] , size[1] , 2)
    grid[: , : , 0] = w.unsqueeze(0).repeat(size[0] , 1)
    grid[: , : , 1] = h.unsqueeze(0).repeat(size[1] , 1).transpose(0 , 1)
    grid = grid.unsqueeze(0).repeat(x.size(0),1,1,1)
    if x.is_cuda:
        grid = grid.cuda()
    return torch.nn.functional.grid_sample(x , grid , mode = mode)

def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


'''
description: 
param {*}
return {*}
'''
class SegList(torch.utils.data.Dataset):
    def __init__(self, data_dir, phase, transforms, list_dir=None,
                 out_name=False):
        self.list_dir = data_dir if list_dir is None else list_dir
        self.data_dir = data_dir
        self.out_name = out_name
        self.phase = phase
        self.transforms = transforms
        self.image_list = None
        self.label_list = None
        self.bbox_list = None
        self.read_lists()

    def __getitem__(self, index):
        data = [Image.open(join(self.data_dir, self.image_list[index]))]
        data = np.array(data[0])
        if len(data.shape) == 2:
            data = np.stack([data , data , data] , axis = 2)
        data = [Image.fromarray(data)]
        if self.label_list is not None:
       	    data.append(Image.open(join(self.data_dir, self.label_list[index])))
        data = list(self.transforms(*data))
        if self.out_name:
            if self.label_list is None:
                data.append(data[0][0, :, :])
            data.append(self.image_list[index])
        return tuple(data)

    def __len__(self):
        return len(self.image_list)

    def read_lists(self):
        image_path = join(self.list_dir, self.phase + FILE_DESCRIPTION+ '_images.txt')
        label_path = join(self.list_dir, self.phase + FILE_DESCRIPTION+ '_labels.txt')
        assert exists(image_path)
        self.image_list = [line.strip() for line in open(image_path, 'r')]
        if exists(label_path):
            self.label_list = [line.strip() for line in open(label_path, 'r')]
            assert len(self.image_list) == len(self.label_list)

'''
description:  将SegMultiHeadList 构建好的数据集整合成一个数据集 , 
param {*}
return {*}
'''
class ConcatSegList(torch.utils.data.Dataset):
    def __init__(self, at, af, seg):
        self.at = at
        self.af = af
        self.seg = seg

    def __getitem__(self, index):
        return (self.at[index], self.af[index], self.seg[index])
    
    def __len__(self):
        return len(self.at)


'''
description:  将SegMultiHeadList 构建好的数据集整合成一个数据集 , 
param {*}
return {*}
'''
class ConcatEDList(torch.utils.data.Dataset):
    def __init__(self, depth, illumination, normal,reflectance):
        self.depth = depth
        self.illumination = illumination
        self.normal = normal
        self.reflectance = reflectance

    def __getitem__(self, index):
        return (self.depth[index], self.illumination[index], self.normal[index],self.reflectance[index])
    
    def __len__(self):
        return len(self.depth)


'''
description: 构建三种数据集
param {*}
return {*}
'''
class SegMultiHeadList(torch.utils.data.Dataset):
    def __init__(self, data_dir, phase, transforms, list_dir=None,
                 out_name=False):
        self.list_dir = data_dir if list_dir is None else list_dir
        self.data_dir = data_dir
        self.out_name = out_name
        self.phase = phase
        self.transforms = transforms
        self.image_list = None
        self.label_list = None
        self.bbox_list = None
        self.read_lists()

    def __getitem__(self, index):
        data = [Image.open(join(self.data_dir, self.image_list[index]))]#* 用PIL 库加载图像 
        data = np.array(data[0]) #? 加载是一张image类,转矩阵,这个很奇怪, 上面故意放一个list里面, 这里又展开
        if len(data.shape) == 2:#? 加载是是标签, 需要进一步处理,这个扩大这个标签是为什么?
            data = np.stack([data , data , data] , axis = 2)
        data = [Image.fromarray(data)]#* 转成一张RGB图像,然后存储到list中, 这个list的第二个标签开始都是放label图像
        
        label_data = list()
        if self.label_list is not None:
            for it in self.label_list[index].split(','):#* 标签是多个图像, 每个图像的相对路径用,分割, 除了semantic 只有一个标签图像,其他都是逗号分隔有多个
       	        label_data.append(Image.open(join(self.data_dir, it))) #* 读取这个标签
            data.append(label_data)#* 放到和图像一个list中
        data = list(self.transforms(*data))#? 不需要判断是否为None吗? ,对图像和标签都进行一样的数据增强
        if self.out_name:#?  加载原来的origin image的相对路径
            if self.label_list is None:
                data.append(data[0][0, :, :])
            data.append(self.image_list[index])
        return tuple(data)

    def __len__(self):
        return len(self.image_list)

    def read_lists(self):
        image_path = join(self.list_dir, self.phase + FILE_DESCRIPTION+ '_images.txt')
        label_path = join(self.list_dir, self.phase + FILE_DESCRIPTION+ '_labels.txt')
        assert exists(image_path)
        self.image_list = [line.strip() for line in open(image_path, 'r')]#*  获取图像相对路径, 相对于 data_dir 的相对路径
        if exists(label_path):
            self.label_list = [line.strip() for line in open(label_path, 'r')]#*  获取标签相对路径, 相对于 data_dir 的相对路径 , 
            assert len(self.image_list) == len(self.label_list)



'''
description: 构建edge detection 数据集的datasets
param {*}
return {*}
'''
class EDMultiHeadList(torch.utils.data.Dataset):
    def __init__(self, data_dir, phase, transforms, list_dir=None,
                 out_name=False):
        self.list_dir = data_dir if list_dir is None else list_dir
        self.data_dir = data_dir
        self.out_name = out_name
        self.phase = phase
        self.transforms = transforms
        self.image_list = None
        self.label_list = None
        self.bbox_list = None
        self.read_lists()

    def __getitem__(self, index):
        data = [Image.open(join(self.data_dir, self.image_list[index]))]
        data = np.array(data[0])
        if len(data.shape) == 2:
            data = np.stack([data , data , data] , axis = 2)
        data = [Image.fromarray(data)]
        
        label_data = list()
        if self.label_list is not None:
            for it in self.label_list[index].split(','):
       	        label_data.append(Image.open(join(self.data_dir, it)))
            data.append(label_data)
        data = list(self.transforms(*data))
        if self.out_name:
            if self.label_list is None:
                data.append(data[0][0, :, :])
            data.append(self.image_list[index])
        return tuple(data)

    def __len__(self):
        return len(self.image_list)

    def read_lists(self):
        image_path = join(self.list_dir, self.phase + FILE_DESCRIPTION+ '_images.txt')
        label_path = join(self.list_dir, self.phase + FILE_DESCRIPTION+ '_labels.txt')
        assert exists(image_path)
        self.image_list = [line.strip() for line in open(image_path, 'r')]
        if exists(label_path):
            self.label_list = [line.strip() for line in open(label_path, 'r')]
            assert len(self.image_list) == len(self.label_list)





'''
description: 
param {*}
return {*}
'''
class SegListMS(torch.utils.data.Dataset):
    def __init__(self, data_dir, phase, transforms, scales, list_dir=None):
        self.list_dir = data_dir if list_dir is None else list_dir
        self.data_dir = data_dir
        self.phase = phase
        self.transforms = transforms
        self.image_list = None
        self.label_list = None
        self.bbox_list = None
        self.read_lists()
        self.scales = scales

    def __getitem__(self, index):
        data = [Image.open(join(self.data_dir, self.image_list[index]))]
        w, h = 640, 480
        data = np.array(data[0])
        if len(data.shape) == 2:
            data = np.stack([data , data , data] , axis = 2)
        data = [Image.fromarray(data)]
        if self.label_list is not None:
            data.append(Image.open(join(self.data_dir, self.label_list[index])))
        out_data = list(self.transforms(*data))
        ms_images = [self.transforms(data[0].resize((round(int(w * s)/32) * 32 , round(int(h * s)/32) * 32),
                                                    Image.BICUBIC))[0]
                     for s in self.scales]
        out_data.append(self.image_list[index])
        out_data.extend(ms_images)
        return tuple(out_data)

    def __len__(self):
        return len(self.image_list)

    def read_lists(self):
        image_path = join(self.list_dir, self.phase + FILE_DESCRIPTION+ '_images.txt')
        label_path = join(self.list_dir, self.phase + FILE_DESCRIPTION+ '_labels.txt')
        assert exists(image_path)
        self.image_list = [line.strip() for line in open(image_path, 'r')]
        if exists(label_path):
            self.label_list = [line.strip() for line in open(label_path, 'r')]
            assert len(self.image_list) == len(self.label_list)

'''
description: 
param {*}
return {*}
'''
class SegListMSMultiHead(torch.utils.data.Dataset):
    def __init__(self, data_dir, phase, transforms, scales, list_dir=None):
        self.list_dir = data_dir if list_dir is None else list_dir
        self.data_dir = data_dir
        self.phase = phase
        self.transforms = transforms
        self.image_list = None
        self.label_list = None
        self.bbox_list = None
        self.read_lists()
        self.scales = scales

    def __getitem__(self, index):
        data = [Image.open(join(self.data_dir, self.image_list[index]))]
        w, h = 640, 480
        data = np.array(data[0])
        if len(data.shape) == 2:
            data = np.stack([data , data , data] , axis = 2)
        data = [Image.fromarray(data)]
        label_data = list()
        if self.label_list is not None:
            for it in self.label_list[index].split(','):
       	        label_data.append(Image.open(join(self.data_dir, it)))
            data.append(label_data)
        out_data = list(self.transforms(*data))
        ms_images = [self.transforms(data[0].resize((round(int(w * s)/32) * 32 , round(int(h * s)/32) * 32),
                                                    Image.BICUBIC))[0]
                     for s in self.scales]
        out_data.append(self.image_list[index])
        out_data.extend(ms_images)
        return tuple(out_data)

    def __len__(self):
        return len(self.image_list)

    def read_lists(self):
        image_path = join(self.list_dir, self.phase + FILE_DESCRIPTION+ '_images.txt')
        label_path = join(self.list_dir, self.phase + FILE_DESCRIPTION+ '_labels.txt')
        assert exists(image_path)
        self.image_list = [line.strip() for line in open(image_path, 'r')]
        if exists(label_path):
            self.label_list = [line.strip() for line in open(label_path, 'r')]
            assert len(self.image_list) == len(self.label_list)


def validate(val_loader, model, criterion, eval_score=None, print_freq=10, transfer_model=None, epoch=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    losses_array = list()
    for it in task_list:
        losses_array.append(AverageMeter())
    score = AverageMeter()

    # switch to evaluate mode
    model.eval()
    if transfer_model is not None:
        transfer_model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        with torch.no_grad():
            input = input.cuda()
            input_var = torch.autograd.Variable(input, volatile=True)
            
            target_var = list()
            for idx in range(len(target)):
                target[idx] = target[idx].cuda(non_blocking=True)
                target_var.append(torch.autograd.Variable(target[idx], volatile=True))
            

            # compute output
            
            if transfer_model is not None:
                _, features = model(input_var)
                output = transfer_model(features)
            elif transfer_model is None:
                output, _ = model(input_var)
            softmaxf = nn.LogSoftmax()

            loss_array = list()
            for idx in range(len(output)):
                output[idx] = softmaxf(output[idx])
                loss_array.append(criterion(output[idx],target_var[idx]))

            loss = sum(loss_array)

            # measure accuracy and record loss

            losses.update(loss.item(), input.size(0))

            for idx, it in enumerate(task_list):
                (losses_array[idx]).update((loss_array[idx]).item(), input.size(0))

            scores_array = list()

            for idx in range(len(output)):
                scores_array.append(eval_score(output[idx], target_var[idx]))
            
            score.update(np.nanmean(scores_array), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            logger.info('Test: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Score {score.val:.3f} ({score.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                score=score))
            
    TENSORBOARD_WRITER.add_scalar('val_loss_average', losses.avg, global_step=epoch)
    TENSORBOARD_WRITER.add_scalar('val_score_average', score.avg, global_step=epoch)

    logger.info(' * Score {top1.avg:.3f}'.format(top1=score))

    return score.avg


'''
description:  
param undefined
param undefined
param undefined
param undefined
param undefined
param undefined
param undefined
return {*}
'''
def validate_cerberus(val_loader, model, criterion, eval_score=None, print_freq=10, transfer_model=None, epoch=None):
    

    #!=======================
    # task_list_array = [['Wood','Painted','Paper','Glass','Brick','Metal','Flat','Plastic','Textured','Glossy','Shiny'],
    #                    ['L','M','R','S','W'],
    #                    ['Segmentation']] 
    task_list_array = [['depth'],
                       ['illumination'],
                       ['normal'],
                       ['reflectance'],]
 

    root_task_list_array = ['depth', 'illumination', 'normal',"reflectance"]
    
    
    
    batch_time_list = list()
    losses_list = list()
    losses_array_list = list()
    score_list = list()
    score = AverageMeter()

    # for i in range(3):
    for i in range(4):
        batch_time_list.append(AverageMeter())
        losses_list.append(AverageMeter())
        losses_array = list()
        for it in task_list_array[i]:
            losses_array.append(AverageMeter())
        losses_array_list.append(losses_array)
        score_list.append(AverageMeter())
    #!=======================

    # switch to evaluate mode
    model.eval()
    # if transfer_model is not None:
    #     transfer_model.eval()

    end = time.time()
    for i, pairs in enumerate(val_loader):
        for index, (input,target) in enumerate(pairs):
            with torch.no_grad():
                input = input.cuda()
                input_var = torch.autograd.Variable(input, volatile=True)
                
                target_var = list()
                for idx in range(len(target)):
                    target[idx] = target[idx].cuda(non_blocking=True)
                    target_var.append(torch.autograd.Variable(target[idx], volatile=True))
                

                # compute output
                output, _, _ = model(input_var, index)
                softmaxf = nn.LogSoftmax()

                loss_array = list()
                for idx in range(len(output)):
                    output[idx]= softmaxf(output[idx])
                    loss_array.append(criterion(output[idx],target_var[idx]))

                loss = sum(loss_array)

                # measure accuracy and record loss

                losses_list[index].update(loss.item(), input.size(0))

                for idx, it in enumerate(task_list_array[index]):
                    (losses_array_list[index][idx]).update((loss_array[idx]).item(), input.size(0))

                scores_array = list()
                #!===========
                # if index < 2:
                if index < 3:
                    for idx in range(len(output)):
                        scores_array.append(eval_score(output[idx], target_var[idx]))
                # elif index == 2:
                elif index == 3:
                    for idx in range(len(output)):
                        scores_array.append(mIoUAll(output[idx], target_var[idx]))
                else:
                    assert 0 == 1
                #!===========
                
                tmp = np.nanmean(scores_array)
                if not np.isnan(tmp):
                    score_list[index].update(tmp, input.size(0))
                else:
                    pass

            # measure elapsed time
            batch_time_list[index].update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                logger.info('Test: [{0}/{1}]\t'
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                            'Score {score.val:.3f} ({score.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time_list[index], loss=losses_list[index],
                    score=score_list[index]))
        #!+================
        # score.update(np.nanmean([score_list[0].val, score_list[1].val, score_list[2].val]))
        score.update(np.nanmean([score_list[0].val, score_list[1].val, score_list[2].val,score_list[3].val]))
        #!+================
        if i % print_freq == 0:
            logger.info('total score is:{score.val:.3f} ({score.avg:.3f})'.format(
                score = score
            ))
    #!===============================
    need_upload = {}
    # for idx, item in enumerate(['attribute','affordance','segmentation']):
    for idx, item in enumerate(root_task_list_array):
        TENSORBOARD_WRITER.add_scalar('val_'+ item +'_loss_average', losses_list[idx].avg, global_step=epoch)
        TENSORBOARD_WRITER.add_scalar('val_'+ item +'_score_average', score_list[idx].avg, global_step=epoch)
        need_upload['val_'+ item +'_loss_average']  = losses_list[idx].avg
        need_upload['val_'+ item +'_score_average']  = score_list[idx].avg
        
    
    logger.info(' * Score {top1.avg:.3f}'.format(top1=score))
    TENSORBOARD_WRITER.add_scalar('val_score_average', score.avg, global_step=epoch)    
    
    need_upload ['val_score_average']=score.avg
    wandb.log(need_upload)
    #!===============================

    return score.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target):
    """Computes the precision@k for the specified values of k"""
    # batch_size = target.size(0) * target.size(1) * target.size(2)
    _, pred = output.max(1)
    pred = pred.view(1, -1)
    target = target.view(1, -1)
    correct = pred.eq(target)
    correct = correct[target != 255]
    correct = correct.view(-1)
    try:
        score = correct.float().sum(0).mul(100.0 / correct.size(0))
        return score.item()
    except:
        return 0

def mIoU(output, target):
    """Computes the iou for the specified values of k"""
    num_classes = output.shape[1]
    hist = np.zeros((num_classes, num_classes))
    _, pred = output.max(1)#* 这个不适合 边缘检测评估 , 因为不知道如何确定这个阈值
    pred = pred.cpu().data.numpy()
    target = target.cpu().data.numpy()
    hist += fast_hist(pred.flatten(), target.flatten(), num_classes)
    ious = per_class_iu(hist) * 100
    return round(np.nanmean(ious[1]), 2)

def mIoUAll(output, target):
    """Computes the iou for the specified values of k"""
    num_classes = output.shape[1]
    hist = np.zeros((num_classes, num_classes))
    _, pred = output.max(1)
    pred = pred.cpu().data.numpy()
    target = target.cpu().data.numpy()
    hist += fast_hist(pred.flatten(), target.flatten(), num_classes)
    ious = per_class_iu(hist) * 100
    return round(np.nanmean(ious), 2)
    

def train(train_loader, model, criterion, optimizer, epoch,
          eval_score=None, print_freq=1, transfer_model=None, transfer_optim=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_array = list()
    for it in task_list:
        losses_array.append(AverageMeter())
    scores = AverageMeter()

    # switch to train mode
    model.train()

    if transfer_model is not None:
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        transfer_model.train()

    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda()
        input_var = torch.autograd.Variable(input)

        target_var = list()
        for idx in range(len(target)):
            target[idx] = target[idx].cuda()
            target_var.append(torch.autograd.Variable(target[idx]))

        # compute output
        if transfer_model is None:
            output, _ = model(input_var)
        elif transfer_model is not None:
            _, features = model(input_var)
            output = transfer_model(features)

        softmaxf = nn.LogSoftmax()
        loss_array = list()

        assert len(output) == len(target)

        for idx in range(len(output)):
            output[idx] = softmaxf(output[idx])
            loss_array.append(criterion(output[idx],target_var[idx]))

        loss = sum(loss_array)

        # measure accuracy and record loss

        losses.update(loss.item(), input.size(0))

        for idx, it in enumerate(task_list):
            (losses_array[idx]).update((loss_array[idx]).item(), input.size(0))

        scores_array = list()

        for idx in range(len(output)):
            scores_array.append(eval_score(output[idx], target_var[idx]))
        
        scores.update(np.nanmean(scores_array), input.size(0))

        # compute gradient and do SGD step
        if transfer_optim is not None:
            transfer_optim.zero_grad()
        elif transfer_optim is None:
            optimizer.zero_grad()

        loss.backward()

        if transfer_optim is not None:
            transfer_optim.step()
        elif transfer_optim is None:
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            losses_info = ''
            for idx, it in enumerate(task_list):
                losses_info += 'Loss_{0} {loss.val:.4f} ({loss.avg:.4f})\t'.format(it, loss=losses_array[idx])
                TENSORBOARD_WRITER.add_scalar('train_task_' + it + '_loss_val', losses_array[idx].val, 
                    global_step= epoch * len(train_loader) + i)
                TENSORBOARD_WRITER.add_scalar('train_task_' + it + '_loss_average', losses_array[idx].avg,
                    global_step= epoch * len(train_loader) + i)

            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        '{loss_info}'
                        'Score {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses,loss_info=losses_info,
                top1=scores))
            
            TENSORBOARD_WRITER.add_scalar('train_loss_val', losses.val, global_step= epoch * len(train_loader) + i)
            TENSORBOARD_WRITER.add_scalar('train_loss_average', losses.avg, global_step= epoch * len(train_loader) + i)
            TENSORBOARD_WRITER.add_scalar('train_scores_val', scores.val, global_step= epoch * len(train_loader) + i)
            TENSORBOARD_WRITER.add_scalar('train_scores_val', scores.avg, global_step= epoch * len(train_loader) + i)

    TENSORBOARD_WRITER.add_scalar('train_epoch_loss_average', losses.avg, global_step= epoch)
    TENSORBOARD_WRITER.add_scalar('train_epochscores_val', scores.avg, global_step= epoch)




'''
description: 
return {*}
'''
def train_cerberus(train_loader, model, criterion, optimizer, epoch,
          eval_score=None, print_freq=1): # transfer_model=None, transfer_optim=None):
    #!==============
    # task_list_array = [['Wood','Painted','Paper','Glass','Brick','Metal','Flat','Plastic','Textured','Glossy','Shiny'],
    #                    ['L','M','R','S','W'],
    #                    ['Segmentation']]

    # root_task_list_array = ['At', 'Af', 'Seg']
    task_list_array = [['depth'],
                       ['illumination'],
                       ['normal'],
                       ['reflectance'],]

    
    root_task_list_array = ['depth', 'illumination', 'normal',"reflectance"]
    #!==============

    batch_time_list = list()
    data_time_list = list()
    losses_list = list()
    losses_array_list = list()
    scores_list = list()
    #!=========
    # for i in range(3):
    for i in range(4):
    #!=========
        batch_time_list.append(AverageMeter())
        data_time_list.append(AverageMeter())
        losses_list.append(AverageMeter())
        losses_array = list()
        for it in task_list_array[i]:
            losses_array.append(AverageMeter())
        losses_array_list.append(losses_array)
        scores_list.append(AverageMeter())

    model.train()

    end = time.time()

    # moo = True
    moo = True
    debug =False

    for i, in_tar_name_pair in enumerate(train_loader):#* 一个一个batch取数据
        if moo :
            grads = {}
        
        if debug:
            task_loss_array_new=[]
            input_new = in_tar_name_pair[0].cuda() #* image
            B,C,W,H=in_tar_name_pair[0].shape
            for index_new, ( target_new) in enumerate(in_tar_name_pair[1][0][1:]):
                target_new=  target_new.reshape(B,W,H)
                input_var_new = torch.autograd.Variable(input_new)
                target_var_new   = list()

                target_var_new= torch.autograd.Variable(target_new.cuda())#* 跑几次还是会遇到上次的问题
                # for idx in range (len(target_new)):
                #     target_var_new.append(torch.autograd.Variable(target_new[idx].cuda()))

                output_new, _, _ = model(input_var_new, index_new)
                
                loss_array_new = [criterion(output_new[idx],target_var_new[idx].reshape(output_new[idx].shape) ) for idx in range(len(output_new))]
                local_loss_new = sum(loss_array_new)
                task_loss_array_new.append(local_loss_new)

            assert len(task_loss_array_new) == 4
            loss_new = sum(task_loss_array_new) 
            all_need_upload= {}
            loss_old ={ "loss_"+k:v  for k,v  in  zip(root_task_list_array,task_loss_array_new)}
            # loss_scale= { "scale_"+k:v  for k,v  in  zip(root_task_list_array,sol)}
            all_need_upload.update(loss_old)
            # all_need_upload.update(loss_scale)
            all_need_upload.update({"total_loss":loss_new})
            wandb.log(all_need_upload)
            
            optimizer.zero_grad()
            loss_new.backward()
            optimizer.step()
        else :
            task_loss_array = []
            in_tar_name_pair_labels = in_tar_name_pair[1].permute(1,0,2,3)
            for index,  target in enumerate(in_tar_name_pair_labels[1:]):# * 遍历一个一个任务
                input   = in_tar_name_pair[0].clone().cuda()
                # measure data loading time #* input 输入图像数据, target 图像的标签, name : input 的相对路径
                B,W,H= target.shape
                target = target.reshape([B,1,W,H])

                data_time_list[index].update(time.time() - end)
                '''
                description: 前向推理每个子任务的数据的循环内, #!第一个moo, 
                #! 主要就是将数据前向推理 然后计算loss ,存储loss,  然后是反向传播,处理梯度 ?, 不是很清楚是如何处理梯度的  
                #? 存储梯度?
                '''
                if moo:
                    input = input.cuda()
                    input_var = torch.autograd.Variable(input)#?将输入转成可导的对象 

                    target_var = [torch.autograd.Variable(target[idx].cuda()) for idx in range(len(target)) ]
        
                    # compute output
                    output, _, _ = model(input_var, index)
                
                    loss_array = list()

                    # assert len(output) == len(target) #? 为什么模型的输出 大小是[11,1,2,512,512]  11表示的是该subtask 的分类类别,而target 是 [1,512,512],而且11 和1也对不上
                    #! 之前的任务是 每个类计算loss, 以[11,1,2,512,512]  为例子, 会循环11次, 输入[2,512,512] 输出[1,512,512] 也就是每个像素是否属于这个像素, 然后和gt[1,512,512]计算loss
                    for idx in range(len(output)):#? 遍历该子任务预测的各个类别mask, 然后我只有一个类别, 并且只输出概率
                        
                        loss_raw = criterion(output[idx],target_var[idx]) #* 计算loss, output 经过softmax 变成0-1 ,但是target_var 确实0-255
                        
                        loss_enhance = loss_raw 
                        #? 这不是一样的吗?  所以说明代码还是基于其他代码  ?
                        if torch.isnan(loss_enhance):
                            print("nan")
                            logger.info('loss_raw is: {0}'.format(loss_raw))
                            logger.info('loss_enhance is: {0}'.format(loss_enhance)) 
                            exit(0) #* 推出程序
                        else:
                            loss_array.append(loss_enhance)

                        local_loss = sum(loss_array)
                        local_loss_enhance = local_loss 
                        
                    
                    # backward for gradient calculate
                    for cnt in model.pretrained.parameters():
                        cnt.grad = None
                    model.scratch.layer1_rn.weight.grad = None
                    model.scratch.layer2_rn.weight.grad = None
                    model.scratch.layer3_rn.weight.grad = None
                    model.scratch.layer4_rn.weight.grad = None
                    local_loss_enhance.backward() #? 反向传播

                    grads[root_task_list_array[index]] = []
                    #? 遍历backbone 的参数是要干嘛?  存储梯度? 
                    for par_name, cnt in model.pretrained.named_parameters():
                        if cnt.grad is not None:
                            grads[root_task_list_array[index]].append(Variable(cnt.grad.data.clone(),requires_grad = False))
                    grads[root_task_list_array[index]].append(Variable(model.scratch.layer1_rn.weight.grad.data.clone(), requires_grad = False))
                    grads[root_task_list_array[index]].append(Variable(model.scratch.layer2_rn.weight.grad.data.clone(), requires_grad = False))
                    grads[root_task_list_array[index]].append(Variable(model.scratch.layer3_rn.weight.grad.data.clone(), requires_grad = False))
                    grads[root_task_list_array[index]].append(Variable(model.scratch.layer4_rn.weight.grad.data.clone(), requires_grad = False))
                    
                else : #* 
                    input = input.cuda()
                    input_var = torch.autograd.Variable(input)#?将输入转成可导的对象 
                    target_var = list()
                    for idx in range(len(target)):
                        target[idx][target[idx]==255] =   1 #* 
                        target_var.append(torch.autograd.Variable(target[idx].cuda()))
        
                    # compute output
                    output, _, _ = model(input_var, index)

                    loss_array = list()
                    assert len(output) == len(target) #? 为什么模型的输出 大小是[11,1,2,512,512]  11表示的是该subtask 的分类类别,而target 是 [1,512,512],而且11 和1也对不上
                    #! 之前的任务是 每个类计算loss, 以[11,1,2,512,512]  为例子, 会循环11次, 输入[2,512,512] 输出[1,512,512] 也就是每个像素是否属于这个像素, 然后和gt[1,512,512]计算loss
                    for idx in range(len(output)):#? 遍历该子任务预测的各个类别mask, 然后我只有一个类别, 并且只输出概率
                        loss_raw = criterion(output[idx],target_var[idx].reshape(output[idx].shape)) #* 计算loss, output 经过softmax 变成0-1 ,但是target_var 确实0-255
                        loss_array.append(loss_raw)
                    task_loss_array.append(sum(loss_array))
                    

                '''
                description: 前向推理每个子任务的数据的循环内, #!第二个moo, 
                #* 计算当前处理的子任务的每个类别的loss 
                param {*} 
                '''
                if moo: 
                    if torch.isnan(local_loss_enhance):
                        print("nan")
                        logger.info('loss_raw is: {0}'.format(local_loss))
                        logger.info('loss_enhance is: {0}'.format(local_loss_enhance))
                        exit(0)
                        # loss_array.append(loss_enhance)
                    else:
                        task_loss_array.append(local_loss_enhance)

                    # measure accuracy and record loss
                    losses_list[index].update(local_loss_enhance.item(), input.size(0))

                    for idx, it in enumerate(task_list_array[index]): #* 遍历当前处理的子任务的每个类别, 每个类别都有一个loss 
                        (losses_array_list[index][idx]).update((loss_array[idx]).item(), input.size(0))

                    if eval_score is not None:
                        scores_array = list()
                        if index < 3:
                            for idx in range(len(output)):
                                scores_array.append(eval_score(output[idx], target_var[idx]))
                        elif index == 3:
                            for idx in range(len(output)):
                                scores_array.append(mIoUAll(output[idx], target_var[idx]))
                        else:
                            assert 0 == 1
                        scores_list[index].update(np.nanmean(scores_array), input.size(0))

                # compute gradient and do SGD step ,#* index =2 就是最后一个subtask 做完了
                if index == 3:     
                    '''
                    description:  在最后一个子任务 , 前向推理结束, 第三个moo
                    #! 将所有子任务在重新前向推理一次, 计算loss , 对应文章的二次求导
                    param {*}
                    return {*}
                    '''
                    if moo:
                        del input, target, input_var, target_var
                        task_loss_array_new = []
                        #! 重新前向传播计算梯度

                        for index_new, target_new in enumerate(in_tar_name_pair_labels[1:]):
                            input_new = in_tar_name_pair[0].clone().cuda() #* image
                            B,W,H =target_new.shape
                            target_new=  target_new.reshape([B,1,W,H])
                            input_var_new = torch.autograd.Variable(input_new)

                            target_var_new = [torch.autograd.Variable(target_new[idx].cuda()) for idx in range (len(target_new)) ]

                            output_new, _, _ = model(input_var_new, index_new)

                            loss_array_new = [criterion(output_new[idx],target_var_new[idx].reshape(output[idx].shape) ) for idx in range(len(output_new))]
                            
                            
                            local_loss_new = sum(loss_array_new)
                            task_loss_array_new.append(local_loss_new)
                
                        
                        assert len(task_loss_array_new) == 4
                        
                        sol, min_norm = MinNormSolver.find_min_norm_element([grads[cnt] for cnt in root_task_list_array])
                        
                        
                        logger.info('scale is: |{0}|\t|{1}|\t|{2}|\t {3}'.format(sol[0], sol[1], sol[2],sol[3]))
                        
                        
                        loss_new = sol[0] * task_loss_array_new[0] + sol[1] * task_loss_array_new[1] \
                            + sol[2] * task_loss_array_new[2] +  sol[3] * task_loss_array_new[3]  

                            
                        all_need_upload= {}
                        loss_old ={ "loss_"+k:v  for k,v  in  zip(root_task_list_array,task_loss_array_new)}
                        # loss_scale= { "scale_"+k:v  for k,v  in  zip(root_task_list_array,sol)}
                        all_need_upload.update(loss_old)
                        # all_need_upload.update(loss_scale)
                        all_need_upload.update({"total_loss_with_scale":loss_new})
                        wandb.log(all_need_upload)
                        
                        optimizer.zero_grad()
                        loss_new.backward()
                        optimizer.step()
                    else:
                        assert len(task_loss_array) == 4
                        loss = sum(task_loss_array)
                        
                        wandb.log({"loss":loss})
                        
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        del input, target, input_var, target_var
                
                '''
                description:  每个子任务的前向推理循环内 , 第四个moo
                主要是为了计算耗时, 以及打印输出
                param {*}
                return {*}
                '''            
                if moo:
                # measure elapsed time
                    batch_time_list[index].update(time.time() - end)
                    end = time.time()
                    if i % print_freq == 0:
                        losses_info = ''
                        for idx, it in enumerate(task_list_array[index]):
                            losses_info += 'Loss_{0} {loss.val:.4f} ({loss.avg:.4f}) \t'.format(it, loss=losses_array_list[index][idx])
                           
                        logger.info('Epoch: [{0}][{1}/{2}]\t'
                                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                                    '{loss_info}'.format(
                            epoch, i, len(train_loader), batch_time=batch_time_list[index],
                            data_time=data_time_list[index], loss=losses_list[index],loss_info=losses_info))
                      
                 
    
    
    '''
    description:  在一个epoch 的训练之外 
    param {*}
    return {*}
    '''
    for i in range(3):
        TENSORBOARD_WRITER.add_scalar('train_epoch_loss_average', losses_list[index].avg, global_step= epoch)
        # TENSORBOARD_WRITER.add_scalar('train_epoch_scores_val', scores_list[index].avg, global_step= epoch)
        #!=============
        wandb.log({"loss_avg_%d"%i: losses_list[i].avg})
        #!=============

        
    


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def train_seg(args):
    batch_size = args.batch_size
    num_workers = args.workers
    crop_size = args.crop_size

    print(' '.join(sys.argv))

    for k, v in args.__dict__.items():
        print(k, ':', v)
    
    if len(task_list) == 1:
        single_model = DPTSegmentationModel(args.classes, backbone="vitb_rn50_384")
    else:
        single_model = DPTSegmentationModelMultiHead(args.classes, task_list, backbone="vitb_rn50_384")
    model = single_model.cuda()

    if args.trans:
        if len(middle_task_list) == 1:
            single_model = DPTSegmentationModel(40, backbone="vitb_rn50_384")
        else:
            single_model = DPTSegmentationModelMultiHead(2, middle_task_list, backbone="vitb_rn50_384")
        model = single_model.cuda()
        model_trans = TransferNet(middle_task_list, task_list)
        model_trans = model_trans.cuda()
        
    criterion = nn.NLLLoss2d(ignore_index=255)

    criterion.cuda()

    # Data loading code
    data_dir = args.data_dir
    info = json.load(open(join(data_dir, 'info.json'), 'r'))
    normalize = transforms.Normalize(mean=info['mean'],
                                     std=info['std'])
    t = []

    if args.random_rotate > 0:
        t.append(transforms.RandomRotateMultiHead(args.random_rotate))
    if args.random_scale > 0:
        t.append(transforms.RandomScaleMultiHead(args.random_scale))
    t.extend([transforms.RandomCropMultiHead(crop_size),
                transforms.RandomHorizontalFlipMultiHead(),
                transforms.ToTensorMultiHead(),
                normalize])
            
    train_loader = torch.utils.data.DataLoader(
        SegMultiHeadList(data_dir, 'train', transforms.Compose(t)),
        batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True
    )

    val_loader = torch.utils.data.DataLoader(
            SegMultiHeadList(data_dir, 'val', transforms.Compose([
            transforms.RandomCropMultiHead(crop_size),
            transforms.ToTensorMultiHead(),
            normalize,
        ])),
        batch_size=1, shuffle=False, num_workers=num_workers,
        pin_memory=True, drop_last=True
    )


    # define loss function (criterion) and pptimizer
    optimizer = torch.optim.SGD(single_model.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    if args.trans:
        trans_optim = torch.optim.SGD(model_trans.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    cudnn.benchmark = True
    best_prec1 = 0
    start_epoch = 0

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            for name, param in checkpoint['state_dict'].items():
                # name = name[7:]
                model.state_dict()[name].copy_(param)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.trans_resume:
        if os.path.isfile(args.trans_resume):
            print("=> loading trans checkpoint '{}'".format(args.trans_resume))
            checkpoint = torch.load(args.trans_resume)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            for name, param in checkpoint['state_dict'].items():
                # name = name[7:]
                model_trans.state_dict()[name].copy_(param)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.trans_resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading pretrained checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained)
            for name, param in checkpoint['state_dict'].items():
                # name = name[7:]
                model.state_dict()[name].copy_(param)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.evaluate:
        validate(val_loader, model, criterion, eval_score=eval(EVAL_METHOD), epoch=0)
        return

    for epoch in range(start_epoch, args.epochs):
        lr = adjust_learning_rate(args, optimizer, epoch)
        logger.info('Epoch: [{0}]\tlr {1:.06f}'.format(epoch, lr))
        # train for one epoch
        if args.trans:
            train(train_loader, model, criterion, optimizer, epoch, 
              eval_score=eval(EVAL_METHOD), transfer_model=model_trans, transfer_optim=trans_optim)
        else:
            train(train_loader, model, criterion, optimizer, epoch,
              eval_score=eval(EVAL_METHOD))

        # evaluate on validation set
        if args.trans:
            prec1 = validate(val_loader, model, criterion,
              eval_score=eval(EVAL_METHOD), transfer_model=model_trans, epoch=epoch)
        else:
            prec1 = validate(val_loader, model, criterion, eval_score=eval(EVAL_METHOD), epoch=epoch)

        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        checkpoint_path = 'checkpoint_latest.pth.tar'
        
        if args.trans:
            checkpoint_path = str(len(middle_task_list)) + 'transfer'+ \
                str(len(task_list))+checkpoint_path
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model_trans.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=checkpoint_path)
            if (epoch + 1) % 10 == 0:
                history_path = str(len(middle_task_list)) + 'transfer'+ \
                    str(len(task_list)) + 'checkpoint_{:03d}.pth.tar'.format(epoch + 1)
                shutil.copyfile(checkpoint_path, history_path)
        else:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=checkpoint_path)
            if (epoch + 1) % 10 == 0:
                history_path = 'checkpoint_{:03d}.pth.tar'.format(epoch + 1)
                shutil.copyfile(checkpoint_path, history_path)

'''
description: 就是将args 的参数上传至wandb
param {*} args
return {*}
'''
def common_init_wandb(args):
    # project_id = run.name #* 获取wandb 随机给项目分配的id 
    for k, v in args.__dict__.items():
        setattr(wandb.config,k,v)
        print(k, ':', v)
    


    
'''
description:  构建模型, 修改了判断是否分布式训练的代码
param {*} args
return {*}
'''
def construct_model(args):

    if args.distributed_train:
        torch.cuda.set_device(args.local_rank) 
        torch.distributed.init_process_group(backend='nccl')
        print(f"local_rank = {args.local_rank}")
        single_model = CerberusSegmentationModelMultiHead(backbone="vitb_rn50_384")
        model = single_model.cuda()
        # model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True) #todo : find_unused_parameters   是干嘛的?
        model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[args.local_rank],
                output_device=args.local_rank) 
        # model = torch.nn.parallel.DataParallel(model,device_ids=[0,1, 2]) #* success 
        # model = torch.nn.parallel.DataParallel(model)  #* success 
        model=model.module
    else :
        single_model = CerberusSegmentationModelMultiHead(backbone="vitb_rn50_384")
        model = single_model.cuda()
    return model,single_model


'''
description:  构建训练数据集的dataloader
param {*} data_dir : 数据集路径
return {*}
'''
def construct_train_data(args):
    # data_dir = args.data_dir   
    train_dataset = Mydataset(root_path="dataset/BSDS-RIND/BSDS-RIND/Augmentation/", split='trainval', crop_size=320)
    

    if args.distributed_train:
        train_sampler = DistributedSampler(train_dataset) # 这个sampler会自动分配数据到各个gpu上
        train_loader = (torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size, num_workers=args.workers,
                pin_memory=True, drop_last=True, sampler=train_sampler
        ))
    else :
        train_loader = (torch.utils.data.DataLoader(
            train_dataset,batch_size=args.batch_size, shuffle=True, 
            num_workers=args.workers,pin_memory=True, drop_last=True, 
        ))

    return train_loader
    

'''
description: 
param {*} data_dir
param {*} args 
return {*}
#todo 将 construct_val_data 和construct_train_data 整合在一起
'''
def  construct_val_data(args):
    data_dir = args.data_dir   
    info = json.load(open(join(data_dir, 'info.json'), 'r'))#*数据集的信息,主要用于构建  normalize , 数据集处理类
    normalize = transforms.Normalize(mean=info['mean'],
                                     std=info['std'])

    phase = "test"

    #* 验证集做的数据预处理比较多,  
    dataset_depth_val = SegMultiHeadList(data_dir, phase+'_depth', transforms.Compose([
                transforms.RandomCropMultiHead(args.crop_size),
                transforms.ToTensorMultiHead(),
                normalize,
            ]))
    dataset_illumination_val = SegMultiHeadList(data_dir, phase+'_illumination', transforms.Compose([
                transforms.RandomCropMultiHead(args.crop_size),
                transforms.ToTensorMultiHead(),
                normalize,
            ]))
    dataset_normal_val = SegMultiHeadList(data_dir, phase+'_normal', transforms.Compose([
                transforms.RandomCropMultiHead(args.crop_size),
                transforms.ToTensorMultiHead(),
                normalize,
            ]))
    dataset_reflection_val = SegMultiHeadList(data_dir, phase+'_reflectance', transforms.Compose([
                transforms.RandomCropMultiHead(args.crop_size),
                transforms.ToTensorMultiHead(),
                normalize,
            ]))

    
    concated_val_datasets = ConcatEDList(dataset_depth_val, dataset_illumination_val, dataset_normal_val,dataset_reflection_val)
    if args.distributed_train: #* 是否分布式训练判断
        val_sampler = DistributedSampler(concated_val_datasets) # 这个sampler会自动分配数据到各个gpu上
       
        val_loader = (torch.utils.data.DataLoader(
            concated_val_datasets,
            batch_size=1, num_workers=args.workers,
            pin_memory=True, drop_last=True,sampler=val_sampler
        ))
    else :
        val_loader = (torch.utils.data.DataLoader(
            concated_val_datasets,
            batch_size=1, shuffle=False, num_workers=args.workers,
            pin_memory=True, drop_last=True
        ))
    return val_loader



'''
description: 
return {*}
'''
class AttentionLoss(nn.Module):
    def __init__(self,alpha=0.1,gamma=2,lamda=0.5):
        super(AttentionLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.lamda = lamda

    '''
    description:  
    param {*} self
    param {*} output
    param {*} label
    return {*}
    '''
    def forward(self,output,label):
        # batch_size, c, height, width = label.size()# B,C,H,W 
        total_loss = 0
        for i in range(len(output)):#* 一个通道一个通道计算loss , 每个通道表示一个类别
            o = output[i] #* [1,H,W]
            l = label[i] #*[C,H,W]
            loss_focal = self.attention_loss2(o, l)#? 
            total_loss = total_loss + loss_focal
        # total_loss = total_loss / batch_size

        return total_loss
    
    '''
    description: 
    param {*} self
    param {*} output
    param {*} target
    return {*}
    '''
    def attention_loss2(self, output,target):
        num_pos = torch.sum(target == 1).float()
        num_neg = torch.sum(target == 0).float()
        alpha = num_neg / (num_pos + num_neg) * 1.0
        eps = 1e-14
        p_clip = torch.clamp(output, min=eps, max=1.0 - eps)

        weight = target * alpha * (4 ** ((1.0 - p_clip) ** 0.5)) + \
                (1.0 - target) * (1.0 - alpha) * (4 ** (p_clip ** 0.5))
        weight=weight.detach()

        #!========
        target = target.float()
        #!========
        loss = F.binary_cross_entropy(output, target, weight, reduction='none')
        loss = torch.sum(loss)
        return loss







'''
description: 
param {*} args
return {*}
'''
def train_seg_cerberus(args):
    common_init_wandb(args) 
    print(' '.join(sys.argv))
    
    model,single_model = construct_model(args)


    #!+=============
    # criterion = nn.NLLLoss2d(ignore_index=255)
    criterion = AttentionLoss()
    criterion.cuda()
    

    #!+=============
    # Data loading code
    
    train_loader = construct_train_data(args)
    # val_loader= construct_val_data(args)
    #*==========
    # define loss function (criterion) and pptimizer
    optimizer = torch.optim.SGD([
                                {'params':single_model.pretrained.parameters()},
                                {'params':single_model.scratch.parameters()}],
                                # {'params':single_model.sigma.parameters(), 'lr': args.lr * 0.01}],
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    cudnn.benchmark = True
    best_prec1 = 0
    start_epoch = 0

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            for name, param in checkpoint['state_dict'].items():
                # name = name[7:]
                model.state_dict()[name].copy_(param)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading pretrained checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained)
            for name, param in checkpoint['state_dict'].items():
                # name = name[7:]
                if name[:5] == 'sigma':
                    model.state_dict()[name].copy_(param)
                else:
                    # model.state_dict()[name].copy_(param)
                    pass
            print("=> loaded sigma checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.pretrained_model:
        if os.path.isfile(args.pretrained_model):
            print("=> loading pretrained checkpoint '{}'".format(args.pretrained_model))
            checkpoint = torch.load(args.pretrained_model)
            for name, param in checkpoint['state_dict'].items():
                if name[:5] == 'sigma':
                    pass
                    # model.state_dict()[name].copy_(param)
                else:
                    model.state_dict()[name].copy_(param)
                    # pass
            print("=> loaded model checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))



    for epoch in range(start_epoch, args.epochs):
        lr = adjust_learning_rate(args, optimizer, epoch)
        logger.info('Epoch: [{0}]\tlr {1:.06f}'.format(epoch, lr))
        #!+=====================
        # train_cerberus(train_loader, model, criterion, optimizer, epoch,
        # eval_score=mIoU)
        
        train_cerberus(train_loader, model, criterion, optimizer, epoch)
        #!+=====================
        
        #if epoch%10==1:
        #!+===========
        # prec1 = validate_cerberus(val_loader, model, criterion, eval_score=mIoU, epoch=epoch)
        # wandb.log({"prec":prec1})

        # is_best = prec1 > best_prec1
        # best_prec1 = max(prec1, best_prec1)
        is_best =True #* 假设每次都是最好的 
        #!+=========== 
        checkpoint_path = 'checkpoint_latest.pth.tar'
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=checkpoint_path)

        if (epoch + 1) % 5 == 0:
            history_path =osp.join("networks/exp1",'checkpoint_{:03d}.pth.tar'.format(epoch + 1)) 
            shutil.copyfile(checkpoint_path, history_path)


def adjust_learning_rate(args, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if args.lr_mode == 'step':
        lr = args.lr * (0.1 ** (epoch // args.step))
    elif args.lr_mode == 'poly':
        lr = args.lr * (1 - epoch / args.epochs) ** 0.9
    else:
        raise ValueError('Unknown lr mode {}'.format(args.lr_mode))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    #adjust the learning rate of sigma
    optimizer.param_groups[-1]['lr'] = lr * 0.01
    
    return lr


def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    return np.bincount(
        n * label[k].astype(int) + pred[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):#? 这是什么 
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

def save_output_images(predictions, filenames, output_dir):
    """
    Saves a given (B x C x H x W) into an image file.
    If given a mini-batch tensor, will save the tensor as a grid of images.
    """
    # pdb.set_trace()
    for ind in range(len(filenames)):
        im = Image.fromarray(predictions[ind].astype(np.uint8))
        fn = os.path.join(output_dir, filenames[ind][:-4] + '.png')
        out_dir = split(fn)[0]
        if not exists(out_dir):
            os.makedirs(out_dir)
        im.save(fn)


def save_colorful_images(predictions, filenames, output_dir, palettes):
   """
   Saves a given (B x C x H x W) into an image file.
   If given a mini-batch tensor, will save the tensor as a grid of images.
   """
   for ind in range(len(filenames)):
       im = Image.fromarray(palettes[predictions[ind].squeeze()])
       fn = os.path.join(output_dir, filenames[ind][:-4] + '.png')
       out_dir = split(fn)[0]
       if not exists(out_dir):
           os.makedirs(out_dir)
       im.save(fn)

def resize_4d_tensor(tensor, width, height):
    tensor_cpu = tensor.cpu().numpy()
    if tensor.size(2) == height and tensor.size(3) == width:
        return tensor_cpu
    out_size = (tensor.size(0), tensor.size(1), height, width)
    out = np.empty(out_size, dtype=np.float32)

    def resize_one(i, j):
        out[i, j] = np.array(
            Image.fromarray(tensor_cpu[i, j]).resize(
                (width, height), Image.BILINEAR))

    def resize_channel(j):
        for i in range(tensor.size(0)):
            out[i, j] = np.array(
                Image.fromarray(tensor_cpu[i, j]).resize(
                    (width, height), Image.BILINEAR))

    workers = [threading.Thread(target=resize_channel, args=(j,))
               for j in range(tensor.size(1))]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    return out


def test_ms(eval_data_loader, model, num_classes, scales,
            output_dir='pred', has_gt=True, save_vis=False):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    hist_array_acc = list()
    hist_array = list()
    iou_compute_cmd = 'per_class_iu(hist_array[idx])'
    if num_classes == 2:
        iou_compute_cmd = '[' + iou_compute_cmd + '[1]]'

    iou_compute_cmd_acc = 'per_class_iu(hist_array_acc[idx])'
    if num_classes == 2:
        iou_compute_cmd_acc = '[' + iou_compute_cmd_acc + '[1]]'

    for i in range(len(task_list)):
        hist_array_acc.append(np.zeros((num_classes, num_classes)))
        hist_array.append(np.zeros((num_classes, num_classes)))

    num_scales = len(scales)
    for itera, input_data in enumerate(eval_data_loader):
        data_time.update(time.time() - end)
        
        if has_gt:
            name = input_data[2]
            label = input_data[1]
        else:
            name = input_data[1]

        logger.info('file name is %s', name)
        
        h, w = input_data[0].size()[2:4]
        images = input_data[-num_scales:]
        outputs = []

        with torch.no_grad():
            for image in images:
                image_var = Variable(image, requires_grad=False)
                image_var = image_var.cuda()
                final, _ = model(image_var)
                final_array = list()
                for entity in final:
                    final_array.append(entity.data)
                outputs.append(final_array)

            final = list()
            for label_idx in range(len(outputs[0])):
                tmp_tensor_list = list()
                for out in outputs:
                    tmp_tensor_list.append(resize_4d_tensor(out[label_idx], w, h))
                
                final.append(sum(tmp_tensor_list))
            pred = list()
            for label_entity in final:
                pred.append(label_entity.argmax(axis=1))

        batch_time.update(time.time() - end)
        if save_vis:
            for idx in range(len(label)):
                assert len(name) == 1
                file_name = (name[0][:-4] + task_list[idx] + '.png',)
                save_output_images(pred[idx], file_name, output_dir)
                save_colorful_images(pred[idx], file_name, output_dir + '_color',
                                    PALETTE)
        if has_gt:
            map_score_array = list()
            for idx in range(len(label)):
                label[idx] = label[idx].numpy()
                hist_array[idx] = fast_hist(pred[idx].flatten(), label[idx].flatten(), num_classes)
                hist_array_acc[idx] += hist_array[idx]

                map_score_array.append(round(np.nanmean(eval(iou_compute_cmd)) * 100, 2))

                logger.info('===> task${}$ mAP {mAP:.3f}'.format(
                    task_list[idx],
                    mAP= map_score_array[idx]))
            
            if len(map_score_array) > 1:
                assert len(map_score_array) == len(label)
                logger.info('===> task${}$ mAP {mAP:.3f}'.format(
                    TASK,
                    mAP= round(np.nanmean(map_score_array),2)))

        end = time.time()
        logger.info('Eval: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    .format(itera, len(eval_data_loader), batch_time=batch_time,
                            data_time=data_time))
    if has_gt: #val
        ious = list()
        for idx in range(len(hist_array_acc)):
            tmp_result = [i * 100.0 for i in eval(iou_compute_cmd_acc)]
            ious.append(tmp_result)
        for idx, i in enumerate(ious):
            logger.info('task %s', task_list[idx])
            logger.info(' '.join('{:.3f}'.format(ii) for ii in i))
        return round(np.nanmean(ious), 2)


def test_ms_cerberus(eval_data_loader, model, scales,
            output_dir='pred', has_gt=True, save_vis=False):
    model.eval()
    
    #!===============
    # task_list_array = [['Wood','Painted','Paper','Glass','Brick','Metal','Flat','Plastic','Textured','Glossy','Shiny'],
    #                    ['L','M','R','S','W'],
    #                    ['Segmentation']]

    # task_name = ['Attribute', 'Affordance', 'Segmentation']


    task_list_array = [['depth'],
                       ['illumination'],
                       ['normal'],
                       ['reflectance']]



    task_name = ['depth', 'illumination', 'normal',"reflectance"]

    #!===============
    
    batch_time_array = list()
    data_time_array = list()
    hist_array_array = list()
    hist_array_array_acc = list()
    # for i in range(3):
    for i in range(4):
        batch_time_array.append(AverageMeter())
        data_time_array.append(AverageMeter())
        hist_array_array.append([])
        hist_array_array_acc.append([])
        
        num_classes = 2
        # if i < 2:
        #     num_classes = 2
        # elif i == 2:
        #     num_classes = 40
        # else:
        #     assert 0 == 1
        
        for j in range(len(task_list_array[i])):
            hist_array_array[i].append(np.zeros((num_classes, num_classes)))
            hist_array_array_acc[i].append(np.zeros((num_classes, num_classes)))



    # batch_time = AverageMeter()
    # data_time = AverageMeter()
    end = time.time()
    #!=========
    # for i, in_tar_pair in enumerate(zip(eval_data_loader[0], eval_data_loader[1], eval_data_loader[2])):
    for i, in_tar_pair in enumerate(zip(eval_data_loader[0], eval_data_loader[1], eval_data_loader[2],eval_data_loader[3])):
    #!=========
        for index, input in enumerate(in_tar_pair):#*一次循环代表的是一次subtask 
            num_classes = 2
            PALETTE = AFFORDANCE_PALETTE
            # if index < 2:
            #     num_classes = 2
            #     PALETTE = AFFORDANCE_PALETTE
            # elif index == 2:
            #     num_classes = 40
            #     PALETTE = NYU40_PALETTE
            # else:
            #     assert 0 == 1
            task_list = task_list_array[index]
            iou_compute_cmd = 'per_class_iu(hist_array_array[index][idx])'

            if num_classes == 2:
            
                iou_compute_cmd = '[' + iou_compute_cmd + '[1]]'

            num_scales = len(scales)
            
            data_time_array[index].update(time.time() - end)
        
            if has_gt:
                name = input[2]
                label = input[1]
            else:
                name = input[1]
            
            logger.info('file name is %s', name)

            h, w = input[0].size()[2:4]
            images = input[-num_scales:]
            outputs = []

            with torch.no_grad():
                for image in images:#? 一个subtask 为什么要用模型重复推理len(scale)次?
                    image_var = Variable(image, requires_grad=False)
                    image_var = image_var.cuda()
               

                    final, _, _ = model(image_var, index)
              

                    final_array = list()
                    for entity in final:
                        final_array.append(entity.data)
                    outputs.append(final_array)

                final = list()
                for label_idx in range(len(outputs[0])):
                    tmp_tensor_list = list()
                    for out in outputs:
                        tmp_tensor_list.append(resize_4d_tensor(out[label_idx], w, h))#* 把结果图像一张一张取出来resize 成 w,h
                    
                    final.append(sum(tmp_tensor_list))#? 4个结果合并成一个!!!, 所以这个scale 指的是将需要测试的图像转成不同的scale进行测试???/
                pred = list()
                for label_entity in final:
                    pred.append(label_entity.argmax(axis=1))#* 根据对每个像素的预测概率得最终预测结果, 也就是从[1,2,W,H] to [1,W,H] ,每个像素点表示类别,要么0要么1

            batch_time_array[index].update(time.time() - end)
            if save_vis:
                for idx in range(len(pred)):
                    assert len(name) == 1
                    file_name = (name[0][:-4] + task_list[idx] + '.png',)
                    save_output_images(pred[idx], file_name, output_dir)
                    save_colorful_images(pred[idx], file_name, output_dir + '_color',
                                        PALETTE)
                    if index == 2:
                        gt_name = (name[0][:-4] + task_list[idx] + '_gt.png',)    
                        label_mask =  (label[idx]==255)

                        save_colorful_images((label[idx]-label_mask*255).numpy(), gt_name, output_dir + '_color',PALETTE)

            if has_gt:#* 有gt就可以计算精度,这个代码和上面代码重复了
                map_score_array = list()
                for idx in range(len(label)):
                    label[idx] = label[idx].numpy()
                    hist_array_array[index][idx] = fast_hist(pred[idx].flatten(),
                     label[idx].flatten(), num_classes)
                    hist_array_array_acc[index][idx] += hist_array_array[index][idx]
                    
                    map_score_array.append(round(np.nanmean([it * 100.0 for it in eval(iou_compute_cmd)]), 2))

                    logger.info('===> task${}$ mAP {mAP:.3f}'.format(
                        task_list[idx],
                        mAP=map_score_array[idx]))
                
                if len(map_score_array) > 1:
                    assert len(map_score_array) == len(label)
                    logger.info('===> task${}$ mAP {mAP:.3f}'.format(
                        task_name[index],
                        mAP=round(np.nanmean(map_score_array),2)))
                        

            end = time.time()
            logger.info('Eval: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        .format(i, len(eval_data_loader[index]), batch_time=batch_time_array[index],
                                data_time=data_time_array[index]))
    if has_gt: #val
        ious_array = list()
        for index, iter in enumerate(hist_array_array_acc):
            ious = list()
            for idx, jter in enumerate(iter):
                iou_compute_cmd = 'per_class_iu(hist_array_array_acc[index][idx])'
                if index < 2:
                    iou_compute_cmd = '[' + iou_compute_cmd + '[1]]'
                tmp_result = [i * 100.0 for i in eval(iou_compute_cmd)]
                ious.append(tmp_result)
            ious_array.append(ious)
        # task_name = ['attribute', 'affordance','segmentation']
        task_name = ['depth', 'illumination', 'normal',"reflectance"]
        for num, ious in enumerate(ious_array):
            for idx, i in enumerate(ious):
                logger.info('task %s', task_list_array[num][idx])#* 输出任务名字
                logger.info(' '.join('{:.3f}'.format(ii) for ii in i))
        for num, ious in enumerate(ious_array):#?
            logger.info('task %s : %.2f',task_name[num],#* 输出任务名字 
                [np.nanmean(i) for i in ious_array][num])

        return round(np.nanmean([np.nanmean(i) for i in ious_array]), 2)


'''
description:  这个函数没有被调用????? #? 为什么
param {*} args
return {*}
'''
def test_seg(args):
    batch_size = args.batch_size
    num_workers = args.workers
    phase = args.phase

    for k, v in args.__dict__.items():
        print(k, ':', v)

    if len(task_list) == 1:
        single_model = DPTSegmentationModel(args.classes, backbone="vitb_rn50_384")
    else:
        single_model = DPTSegmentationModelMultiHead(args.classes, task_list, backbone="vitb_rn50_384")

    checkpoint = torch.load(args.resume)
    
    for name, param in checkpoint['state_dict'].items():
        # name = name[7:]
        single_model.state_dict()[name].copy_(param)
    
    if args.pretrained:
        single_model.load_state_dict(torch.load(args.pretrained))
    model = single_model.cuda()

    data_dir = args.data_dir
    info = json.load(open(join(data_dir, 'info.json'), 'r'))
    normalize = transforms.Normalize(mean=info['mean'], std=info['std'])
    scales = [0.9, 1, 1.25]

    if args.ms:
        dataset = SegListMSMultiHead(data_dir, phase, transforms.Compose([
            transforms.ToTensorMultiHead(),
            normalize,
        ]), scales)
    else:
        dataset = SegMultiHeadList(data_dir, phase, transforms.Compose([
            transforms.ToTensorMultiHead(),
            normalize,
        ]), out_name=True)
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=False
    )

    cudnn.benchmark = True

    # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            for name, param in checkpoint['state_dict'].items():
                # name = name[7:]
                model.state_dict()[name].copy_(param)
            # model.load_state_dict(checkpoint['state_dict'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    out_dir = '{}_{:03d}_{}'.format(args.arch, start_epoch, phase)
    if len(args.test_suffix) > 0:
        out_dir += '_' + args.test_suffix
    if args.ms:
        out_dir += '_ms'

    if args.ms:
        mAP = test_ms(test_loader, model, args.classes, save_vis=True,
                      has_gt=phase != 'test' or args.with_gt,
                      output_dir=out_dir,
                      scales=scales)

    logger.info('%s mAP: %f', TASK, mAP)


'''
description:  这个才是测试函数, 有调用
param {*} args
return {*}
'''
def test_seg_cerberus(args):
    batch_size = args.batch_size
    num_workers = args.workers
    phase = args.phase

    
    # task_list_array = [['Wood','Painted','Paper','Glass','Brick','Metal','Flat','Plastic','Textured','Glossy','Shiny'],#! attribute
    #                 ['L','M','R','S','W'],#! affordence 
    #                 ['Segmentation']]  #! segment class 

    task_list_array = [['depth'],
                       ['illumination'],
                       ['normal'],
                       ['reflectance'],]




    for k, v in args.__dict__.items():
        print(k, ':', v)

    #* 加载模型
    single_model = CerberusSegmentationModelMultiHead(backbone="vitb_rn50_384")

    

    checkpoint = torch.load(args.resume)
    for name, param in checkpoint['state_dict'].items():
        # name = name[7:]
        single_model.state_dict()[name].copy_(param)
    
    model = single_model.cuda()
    
    logger.info(model)
    #* 加载数据
    data_dir = args.data_dir
    info = json.load(open(join(data_dir, 'info.json'), 'r'))
    normalize = transforms.Normalize(mean=info['mean'], std=info['std'])
    #!=========
    scales = [0.9, 1, 1.25]#? 这个scale 什么意思? 
    # scales = [0.9, 1, 1.25,1]#? 这个scale 什么意思? 
    #!=========

    test_loader_list = []
    if args.ms:
        #!=====================
        # for i in ['_attribute', '_affordance', '']:
        for i in ['_depth', '_illumination', '_normal','_reflectance']:
        #!=====================
            #! 分别加载三个子任务的数据
            test_loader_list.append(torch.utils.data.DataLoader(
                SegListMSMultiHead(data_dir, phase + i, transforms.Compose([
                    transforms.ToTensorMultiHead(),
                    normalize,]), 
                scales
                ),
                batch_size=batch_size, shuffle=False, num_workers=num_workers,
                pin_memory=False
            ))
    else:
        assert 0 == 1


    cudnn.benchmark = True
    #? 还是加载模型??
    # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            for name, param in checkpoint['state_dict'].items():
                # name = name[7:]
                model.state_dict()[name].copy_(param)
            # model.load_state_dict(checkpoint['state_dict'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    out_dir = '{}_{:03d}_{}'.format(args.arch, start_epoch, phase)
    if len(args.test_suffix) > 0:
        out_dir += '_' + args.test_suffix
    if args.ms:
        out_dir += '_ms'
    #! 测试吗? 
    if args.ms:
        mAP = test_ms_cerberus(test_loader_list, model, save_vis=True,
                    #   has_gt=phase != 'test' or args.with_gt,
                      has_gt=phase == 'test' or args.with_gt,
                      output_dir=out_dir,
                      scales=scales)
    else:
        assert 0 == 1, 'please add the argument --ms'
    if mAP is not None:
        logger.info('%s mAP: %f', 'average mAP is: ', mAP)



def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='')
    #!=================
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--distributed_train",action='store_true')
    #!=================
    parser.add_argument('cmd', choices=['train', 'test'])
    parser.add_argument('-d', '--data-dir', default='./dataset/BSDS_RIND_mine')
    parser.add_argument('-c', '--classes', default=0, type=int)
    parser.add_argument('-s', '--crop-size', default=0, type=int)
    parser.add_argument('--step', type=int, default=200)
    parser.add_argument('--arch',type=str, default="test_arch",help='save_name dir ')
    # parser.add_argument('--arch')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--lr-mode', type=str, default='step')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('-e', '--evaluate', dest='evaluate',
                        action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--trans-resume', default='', type=str, metavar='PATH',
                        help='path to latest trans checkpoint (default: none)')
    parser.add_argument('--pretrained', dest='pretrained',
                        default='', type=str, metavar='PATH',
                        help='use pre-trained model')
    parser.add_argument('--pretrained-model', dest='pretrained_model',
                        default='', type=str, metavar='PATH',
                        help='use pre-trained model')
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--load-release', dest='load_rel', default=None)
    parser.add_argument('--phase', default='val')
    parser.add_argument('--random-scale', default=0, type=float)
    parser.add_argument('--random-rotate', default=0, type=int)
    parser.add_argument('--bn-sync', action='store_true')
    parser.add_argument('--ms', action='store_true',
                        help='Turn on multi-scale testing')
                        
                    
    parser.add_argument('--trans', action='store_true',
                        help='Turn on transfer learning')
    parser.add_argument('--with-gt', action='store_true')
    parser.add_argument('--test-suffix', default='', type=str)
    args = parser.parse_args()

    assert args.data_dir is not None
    # assert args.classes > 0

    print(' '.join(sys.argv))
    print(args)

    if args.bn_sync:
        drn.BatchNorm = batchnormsync.BatchNormSync

    return args



def main():
    args = parse_args()
    if args.cmd == 'train':
        train_seg_cerberus(args)
    elif args.cmd == 'test':
        test_seg_cerberus(args)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    main()
    torch.cuda.empty_cache()
