'''
Author:   "  "
Date: 2022-06-11 22:47:30
LastEditTime: 2023-08-06 21:47:00
LastEditors: daniel
Description: 
FilePath: /Cerberus-main/utils/utils.py
email:  
'''
import torch
from PIL import Image
import math
from os.path import  join, split
import threading
import shutil
import numpy as np
import os
import torch.nn.functional as F
import argparse
# import drn
from concurrent import futures

try:
    from modules import batchnormsync
except ImportError:
    pass



import scipy.io as scio
import sys



def load_mat(path):
    return scio.loadmat(path)



"""
come from the dataloders.prediction_loaders.base_loader
"""
def load_mat_gt(data_path):
    gt  = load_mat(data_path)
    
    return gt['groundTruth'][0,0]['Boundaries'][0,0]
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
description:  计算可以学习的模型参数
param {*} model : 需要计算的模型
return {*}
'''

def calculate_param_num(model):
    total_params = 0
    Trainable_params = 0
    NonTrainable_params = 0
    for param in model.parameters():
        multvalue = np.prod(param.size())
        total_params += multvalue
        if param.requires_grad:
            Trainable_params += multvalue  # 可训练参数量
        else:
            NonTrainable_params += multvalue  # 非可训练参数量
    
    
    return total_params,Trainable_params,NonTrainable_params











# Print iterations progress (thanks StackOverflow)
def printProgress(iteration, total, prefix='', suffix='', decimals=1, barLength=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        barLength   - Optional  : character length of bar (Int)
    """
    formatStr       = "{0:." + str(decimals) + "f}"
    percents        = formatStr.format(100 * (iteration / float(total)))
    filledLength    = int(round(barLength * iteration / float(total)))
    bar             = '' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\x1b[2K\r')
    sys.stdout.flush()


def process_mp(function,function_parameter_list,num_threads=64,\
                prefix='processing with multiple threads:',suffix = "done"):

    num_sample = len(function_parameter_list)
    with futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
        
        fs = [executor.submit(function, parameters) for parameters in function_parameter_list]

        for i, f in enumerate(futures.as_completed(fs)):
            printProgress(i, num_sample, prefix=prefix, suffix=suffix, barLength=40)







#* from NYUD2 dataloder: 


''' 
description:  Expand non-zero elements of goal by a factor of 10.
param {*} goal 
param {*} times
return {*}
'''
def dilation(goal, times = 2 ):
    selem = skimage.morphology.disk(times)


    # goal = skimage.morphology.binary_dilation(goal, selem) != True
    goal = morphology.binary_dilation(goal, selem) != True
    goal = 1 - goal * 1.
    goal*=255
    return goal



def crop_save(path,src):
        if not exists(path):
            imwrite(path,src[45:471, 41:601])


'''
description:  save  mat file for evaluation 
param {*} self
param {*} file_name
param {*} gt_map
return {*}
'''
def save_as_mat(file_name,gt_map):

    scio.savemat(file_name,
        {'__header__': b'MATLAB 5.0 MAT-file, Platform: MACI64, Created on: Mon Feb 7 06:47:01 2023',
        '__version__': '1.0',
        '__globals__': [],
        'groundTruth': [{'Boundaries':gt_map}]
        }
    )




def readtxt(path):
    return np.loadtxt(path,dtype=np.str0,delimiter='\n')
