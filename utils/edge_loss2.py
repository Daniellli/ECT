'''
Author: xushaocong
Date: 2022-06-07 22:24:07
LastEditTime: 2022-06-07 22:54:53
LastEditors: xushaocong
Description: 
FilePath: /Cerberus-main/utils/edge_loss2.py
email: xushaocong@stu.xmu.edu.cn
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import dice
from IPython import embed

def clip_by_value(t, t_min, t_max):
    result = (t >= t_min)* t + (t < t_min) * t_min
    result = (result <= t_max) * result + (result > t_max)* t_max
    return result

'''
description: 
param undefined
param undefined
return {*}
'''
def attention_loss2(output,target):
    num_pos = torch.sum(target == 1).float()
    num_neg = torch.sum(target == 0).float()
    alpha = num_neg / (num_pos + num_neg) * 1.0
    eps = 1e-14
    p_clip = torch.clamp(output, min=eps, max=1.0 - eps)

    weight = target * alpha * (4 ** ((1.0 - p_clip) ** 0.5)) + \
             (1.0 - target) * (1.0 - alpha) * (4 ** (p_clip ** 0.5))
    weight=weight.detach()
    loss = F.binary_cross_entropy(output, target, weight, reduction='none')
    loss = torch.sum(loss)
    return loss


'''
description: 
return {*}
'''
class AttentionLoss2(nn.Module):
    def __init__(self,alpha=0.1,gamma=2,lamda=0.5):
        super(AttentionLoss2, self).__init__()
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
        batch_size, c, height, width = label.size()# B,C,H,W 
        total_loss = 0
        #!+===========
        # label = label.reshape([len(output),height,width])
        #!+===========
        for i in range(len(output)):
            #!+==============
            o = output[i]#* [1,H,W]
            # o = output[i].reshape(height,width) #* [1,H,W]
            l = label[i] #*[C,H,W]
            #!+==============
            loss_focal = attention_loss2(o, l)#? 
            total_loss = total_loss + loss_focal
        total_loss = total_loss / batch_size
        return total_loss


class AttentionLossSingleMap(nn.Module):
    def __init__(self,alpha=0.1,gamma=2,lamda=0.5):
        super(AttentionLossSingleMap, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.lamda = lamda

    def forward(self,output,label):
        batch_size, c, height, width = label.size()
        loss_focal = attention_loss2(output, label)
        total_loss = loss_focal / batch_size
        return total_loss


if __name__ == '__main__':
    N = 4
    H, W = 320, 320
    label = torch.randint(0, 2, size=(N, 1, H, W)).float()
    o_b = [torch.rand(N, 1, H, W), torch.rand(N, 1, H, W), torch.rand(N, 1, H, W), torch.rand(N, 1, H, W)]
    crientation = AttentionLoss2()
    total_loss = crientation(o_b, label)
    print('loss 2-1 :   '+ str(total_loss))





