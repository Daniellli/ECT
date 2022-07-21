'''
Author: xushaocong
Date: 2022-07-21 11:59:44
LastEditTime: 2022-07-21 12:51:20
LastEditors: xushaocong
Description: 
FilePath: /cerberus/model/loss/inverse_loss.py
email: xushaocong@stu.xmu.edu.cn
'''



import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os 
from .InverseForm import InverseNet
INVERSEFORM_MODULE = os.path.join("pretrained_models", "distance_measures_regressor.pth")

class InverseTransform2D(nn.Module):
    def __init__(self, model_output=None):
        super(InverseTransform2D, self).__init__()
        ## Setting up loss
        self.tile_factor = 3
        self.resized_dim = 672
        self.tiled_dim = self.resized_dim//self.tile_factor
        
        inversenet_backbone = InverseNet()
        self.inversenet = load_model_from_dict(inversenet_backbone, INVERSEFORM_MODULE).cuda()
        for param in self.inversenet.parameters():
            param.requires_grad = False            

    def forward(self, inputs, targets):   
        inputs = F.log_softmax(inputs) 
            
        inputs = F.interpolate(inputs, size=(self.resized_dim, 2*self.resized_dim), mode='bilinear')
        targets = F.interpolate(targets, size=(self.resized_dim, 2*self.resized_dim), mode='bilinear')
        
        batch_size = inputs.shape[0]

        tiled_inputs = inputs[:,:,:self.tiled_dim,:self.tiled_dim]
        tiled_targets = targets[:,:,:self.tiled_dim,:self.tiled_dim]
        k=1      
        for i in range(0, self.tile_factor):
            for j in range(0, 2*self.tile_factor):
                if i+j!=0:
                    tiled_targets = \
                    torch.cat((tiled_targets, targets[:, :, self.tiled_dim*i:self.tiled_dim*(i+1), self.tiled_dim*j:self.tiled_dim*(j+1)]), dim=0)
                    k += 1

        k=1      
        for i in range(0, self.tile_factor):
            for j in range(0, 2*self.tile_factor):
                if i+j!=0:
                    tiled_inputs = \
                    torch.cat((tiled_inputs, inputs[:, :, self.tiled_dim*i:self.tiled_dim*(i+1), self.tiled_dim*j:self.tiled_dim*(j+1)]), dim=0)
                k += 1
                        
        _, _, distance_coeffs = self.inversenet(tiled_inputs, tiled_targets)
        
        mean_square_inverse_loss = (((distance_coeffs*distance_coeffs).sum(dim=1))**0.5).mean()
        return mean_square_inverse_loss



def load_model_from_dict(model, pretrained):
    pretrained_dict = torch.load(pretrained, map_location=torch.device('cpu'))
    model_dict = model.state_dict()
    updated_model_dict = {}
    for k_model, v_model in model_dict.items():
        if k_model.startswith('model') or k_model.startswith('module'):
            k_updated = '.'.join(k_model.split('.')[1:])
            updated_model_dict[k_updated] = k_model
        else:
            updated_model_dict[k_model] = k_model

    updated_pretrained_dict = {}
    for k, v in pretrained_dict.items():
        if k.startswith('model') or k.startswith('modules'):
            k = '.'.join(k.split('.')[1:])
        if k in updated_model_dict.keys() and model_dict[k].shape==v.shape:
            updated_pretrained_dict[updated_model_dict[k]] = v

    model_dict.update(updated_pretrained_dict)
    model.load_state_dict(model_dict)
    return model
    
    