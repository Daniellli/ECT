'''
Author: xushaocong
Date: 2022-06-07 22:24:07
LastEditTime: 2023-03-05 08:23:52
LastEditors: daniel
Description: 
FilePath: /cerberus/utils/edge_loss2.py
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
description:  #? why sum ?
param undefined
param undefined
return {*}
'''
def attention_loss2(output,target,beta=4,gamma = 0.5):
    num_pos = torch.sum(target == 1).float()
    num_neg = torch.sum(target == 0).float()
    alpha = num_neg / (num_pos + num_neg) * 1.0#* 对应loss 公式的alpha ,正负样本均衡的作用
    eps = 1e-14 
    p_clip = torch.clamp(output, min=eps, max=1.0 - eps) #* map  the output into the range [min,max]
    weight = target * alpha * (beta ** ((1.0 - p_clip) ** gamma)) + \
             (1.0 - target) * (1.0 - alpha) * (beta ** (p_clip ** gamma))

    weight=weight.detach()
    loss = F.binary_cross_entropy(output, target, weight, reduction='none')
    loss = torch.sum(loss)
    return loss


'''
description:  for match the scale of inverform loss and semantic edge loss
param undefined
param undefined
return {*}
'''
def attention_loss(output,target,beta=4,gamma = 0.5):
    num_pos = torch.sum(target == 1).float()
    num_neg = torch.sum(target == 0).float()
    alpha = num_neg / (num_pos + num_neg) * 1.0#* 对应loss 公式的alpha ,正负样本均衡的作用
    eps = 1e-14 
    p_clip = torch.clamp(output, min=eps, max=1.0 - eps) #* map  the output into the range [min,max]
    weight = target * alpha * (beta ** ((1.0 - p_clip) ** gamma)) + \
             (1.0 - target) * (1.0 - alpha) * (beta ** (p_clip ** gamma))

    weight=weight.detach()
    loss = F.binary_cross_entropy(output, target, weight, reduction='none')
    loss = torch.mean(loss)
    return loss



'''
description:  for match the scale of inverform loss and semantic edge loss
param undefined
param undefined
return {*}
'''
def attention_loss_with_pad(output,target,padding,beta=4,gamma = 0.5):
    target = torch.mul(target, padding)
    
    num_pos = torch.sum(target == 1).float()
    num_total = torch.sum(padding==1).float()
    num_neg = num_total -  num_pos
    
    # num_pos = torch.sum(target == 1).float()
    # num_neg = torch.sum(target == 0).float()

    alpha = num_neg / (num_pos + num_neg) * 1.0#* 对应loss 公式的alpha ,正负样本均衡的作用
    eps = 1e-14 
    p_clip = torch.clamp(output, min=eps, max=1.0 - eps) #* map  the output into the range [min,max]
    weight = target * alpha * (beta ** ((1.0 - p_clip) ** gamma)) + \
             (1.0 - target) * (1.0 - alpha) * (beta ** (p_clip ** gamma))

    weight=weight.detach()
    loss = F.binary_cross_entropy(output, target, weight, reduction='none')
    loss = torch.mean(loss)
    return loss
'''
description: 
return {*}
'''
class AttentionLoss2(nn.Module):
    def __init__(self,gamma=0.5,beta=4):
        super(AttentionLoss2, self).__init__()
        #* by default ,,alpha=0.1,gamma=2,lamda=0.5  
        self.gamma = gamma
        self.beta = beta

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
        for i in range(len(output)):
            o = output[i].reshape(batch_size,height,width) #* [B,H,W]
            l = label[:,i,:,:] #*[C,H,W]

            # loss_focal = attention_loss2(o, l,beta=self.beta,gamma=self.gamma)
            loss_focal = attention_loss(o, l,beta=self.beta,gamma=self.gamma)

            total_loss = total_loss + loss_focal
        total_loss = total_loss / batch_size
        return total_loss




'''
description: 
return {*}
'''
class AttentionLoss3(nn.Module):
    def __init__(self,gamma=0.5,beta=4):
        super(AttentionLoss3, self).__init__()
        #* by default ,,alpha=0.1,gamma=2,lamda=0.5  
        self.gamma = gamma
        self.beta = beta

    '''
    description:  
    param {*} self
    param {*} output
    param {*} label
    return {*}
    '''
    def forward(self,output,label):
        pad = label[:,0,:,:]
        label = label[:,1:,:,:]


        batch_size, c, height, width = label.size()# B,C,H,W 
        total_loss = 0
        

        for i in range(len(output)):
            try : 
                o = output[i].reshape(batch_size,height,width) #* [B,H,W]
                l = label[:,i,:,:] #*[C,H,W]
            except Exception as e :
                embed()
                print(e) 


            # loss_focal = attention_loss2(o, l,beta=self.beta,gamma=self.gamma)
            loss_focal = attention_loss_with_pad(o, l,padding=pad , beta=self.beta,gamma=self.gamma)

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





