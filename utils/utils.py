'''
Author: xushaocong
Date: 2022-06-11 22:47:30
LastEditTime: 2022-06-22 18:51:54
LastEditors: xushaocong
Description: 
FilePath: /cerberus/utils/utils.py
email: xushaocong@stu.xmu.edu.cn
'''
import torch
from PIL import Image
from torch import nn
# from typing import OrderedDict
import json
import math
from os.path import exists, join, split
import threading
import shutil
import numpy as np
from loguru import logger
import os
# from torch.nn.modules import transformer
# import torch.optim as optim
# import torch.nn.functional as F
import argparse
import drn
import sys
try:
    from modules import batchnormsync
except ImportError:
    pass


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




'''
description: 
param {*} output
param {*} target
return {*}
'''
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




'''
description: 
param {*} x
param {*} size
param {*} scale
param {*} mode
return {*}
'''
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

'''
description:  
param {*} up
return {*}
'''
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
param {*} predictions
param {*} filenames
param {*} output_dir
param {*} palettes
return {*}
'''
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



'''
description: 
param {*} tensor
param {*} width
param {*} height
return {*}
'''
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


'''
description: 
param {*} depth_output_dir
return {*}
'''
def make_dir(depth_output_dir):
    if not os.path.exists(depth_output_dir):
        os.makedirs(depth_output_dir)


'''
description: 
param {*} pred
param {*} label
param {*} n
return {*}
'''
def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    return np.bincount(
        n * label[k].astype(int) + pred[k], minlength=n ** 2).reshape(n, n)




'''
description: 
param {*} hist
return {*}
'''
def per_class_iu(hist):#? 这是什么 
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


'''
description:  保存图像
param {*} predictions
param {*} filenames
param {*} output_dir
return {*}
'''
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




'''
description:   保存model
param {*} state
param {*} is_best
param {*} filename
return {*}
'''
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    #* 在save 一份
    if is_best:
        shutil.copyfile(filename, join(split(filename)[-2] ,'model_best.pth.tar'))



'''
description:  parse args
return {*}
'''
def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('cmd', choices=['train', 'test'])
    parser.add_argument('-d', '--data-dir', default='./dataset/BSDS_RIND_mine')
    parser.add_argument('-s', '--crop-size', default=320, type=int)
    parser.add_argument('--step', type=int, default=200)
    parser.add_argument('--arch',type=str, default="test_arch",help='save_name dir ')
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--lr-mode', type=str, default='poly')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    #todo  : eval during train 
    parser.add_argument('-e', '--evaluate', dest='evaluate',
                        action='store_true',help='evaluate model on validation set')

    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    parser.add_argument('--pretrained-model', dest='pretrained_model',
                        default='', type=str, metavar='PATH',
                        help='use pre-trained model')

    parser.add_argument('-j', '--workers', type=int, default=10)
    parser.add_argument('--bn-sync', action='store_true')#* 暂时没用
    parser.add_argument('--gpu-ids', default='7', type=str)
    parser.add_argument('--moo', action='store_true',
                        help='Turn on multi-objective optimization')
    parser.add_argument("--local_rank", type=int,default=-1,help="node rank for distrubuted training")
    parser.add_argument('--cudnn_benchmark', type=bool, default=True,
                        help='Set cudnn benchmark on (1) or off (0) (default is on).')

    parser.add_argument("--run-id", type=int,default=None,help="for evaluation ")
    parser.add_argument("--bg-weight", type=float,default=1,help=" background weight  ")
    parser.add_argument("--rind-weight", type=float,default=1,help=" rind weight  ")

    parser.add_argument("--train-dir",type=str,default="dataset/BSDS-RIND/BSDS-RIND/Augmentation/",
                help="训练数据集的文件夹root")
    parser.add_argument("--test-dir",type=str,default="dataset/BSDS-RIND/BSDS-RIND/Augmentation/",
                help="训练数据集的文件夹root")
        
    args = parser.parse_args()
    if args.bn_sync:
        drn.BatchNorm = batchnormsync.BatchNormSync

    return args




