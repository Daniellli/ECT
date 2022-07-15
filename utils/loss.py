'''
Author: xushaocong
Date: 2022-06-07 19:26:49
LastEditTime: 2022-07-15 10:40:08
LastEditors: xushaocong
Description: 
FilePath: /cerberus/utils/loss.py
email: xushaocong@stu.xmu.edu.cn
'''

'''
Author: xushaocong
Date: 2022-05-19 22:32:30
LastEditTime: 2022-06-07 19:16:47
LastEditors: xushaocong
Description: 
FilePath: /RINDNet-main/utils/loss.py
email: xushaocong@stu.xmu.edu.cn
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed


def clip_by_value(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    result = (t >= t_min)* t + (t < t_min) * t_min
    result = (result <= t_max) * result + (result > t_max)* t_max
    return result

'''
description: 
return {*}
'''
class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss #! 
        elif mode == 'focal2':
            return self.FocalLoss2
        elif mode == 'attention':
            print('attention loss')
            return self.AttentionLoss
        else:
            raise NotImplementedError

  
    '''
    description:  根据focal loss的公式编写
    param {*} self
    param {*} logit
    param {*} target
    param {*} gamma : focal parameter
    param {*} alpha 样本均衡参数
    return {*}
    '''
    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        #! ignore_index=self.ignore_index 忽略了gt label == 255的数据
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index, 
                                        reduction='sum')
        if self.cuda:
            criterion = criterion.cuda()

        
        logpt = -criterion(logit, target.long())#* 两个维度不一样的数据如何做交叉熵? 
        pt = torch.exp(logpt)  
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt
        loss /= n
        return loss


if __name__ == "__main__":
    loss = SegmentationLosses()

    logit = torch.rand(8, 4, 7, 7)
    target = torch.rand(8, 7, 7)

   # a = a.view(8,196)

    print(loss.FocalLoss(logit, target, gamma=2, alpha=0.5).item())




