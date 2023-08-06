'''
Author:   "  "
Date: 2022-07-21 11:59:44
LastEditTime: 2023-08-06 22:01:10
LastEditors: daniel
Description: 
FilePath: /Cerberus-main/model/loss/inverse_loss.py
email:  
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



    #* cityscape: 1024x2048  , setting :  resize to  672×1344 and  tile it to 18 tiles of 224X224 pixels 
    #* NYU-Depth-v2 and PASCAL , setting :  resize to  672* 672 and  tile it to 9 tiles of 224X224 pixels , 
    # def forward(self, inputs, targets):   
    #     inputs = F.log_softmax(inputs) #* softmax first then log
            
    #     inputs = F.interpolate(inputs, size=(self.resized_dim, self.resized_dim), mode='bilinear') #* from [B,C,320,320] to [B,C,672,1344]
    #     targets = F.interpolate(targets, size=(self.resized_dim, self.resized_dim), mode='bilinear')#* from [B,C,320,320] to [B,C,672,1344]
        
    #     batch_size = inputs.shape[0]

    #     tiled_inputs = inputs[:,:,:self.tiled_dim,:self.tiled_dim] #* pick up [B,C,224,244]
    #     tiled_targets = targets[:,:,:self.tiled_dim,:self.tiled_dim]#* pick up [B,C,224,244]


    #     #* from [1,1,224,244]  to [18,1,224,244], extracting context patch  one by one 
    #     k=1
    #     for i in range(0, self.tile_factor):
    #         for j in range(0, self.tile_factor):
    #             if i+j!=0:
    #                 tiled_targets = \
    #                 torch.cat((tiled_targets, targets[:, :, self.tiled_dim*i:self.tiled_dim*(i+1), self.tiled_dim*j:self.tiled_dim*(j+1)]), dim=0)
    #                 k += 1


    #     #* from [1,1,224,244]  to [18,1,224,244]
    #     k=1      
    #     for i in range(0, self.tile_factor):
    #         for j in range(0, self.tile_factor):
    #             if i+j!=0:
    #                 tiled_inputs = \
    #                 torch.cat((tiled_inputs, inputs[:, :, self.tiled_dim*i:self.tiled_dim*(i+1), self.tiled_dim*j:self.tiled_dim*(j+1)]), dim=0)
    #             k += 1

    #     #* feed forward compute homography 
    #     _, _, distance_coeffs = self.inversenet(tiled_inputs, tiled_targets)

        
    #     mean_square_inverse_loss = (((distance_coeffs*distance_coeffs).sum(dim=1))**0.5).mean() 
    """
    Square each element of the [18 * 4] matrix. Sum the squared elements to obtain a [18] vector. 
    Take the square root of each element in the [18] vector. 
    Calculate the average of the elements to obtain a [1] scalar.
    
    """
    #     return mean_square_inverse_loss


        
    def forward(self, inputs, targets):   
        inputs = F.log_softmax(inputs) #* 先softmax 后取对数 
            
        inputs = F.interpolate(inputs, size=(self.resized_dim, 2*self.resized_dim), mode='bilinear') #* from [B,C,320,320] to [B,C,672,1344]
        targets = F.interpolate(targets, size=(self.resized_dim, 2*self.resized_dim), mode='bilinear')#* from [B,C,320,320] to [B,C,672,1344]
        
        batch_size = inputs.shape[0]

        tiled_inputs = inputs[:,:,:self.tiled_dim,:self.tiled_dim] #* pick up [B,C,224,244]
        tiled_targets = targets[:,:,:self.tiled_dim,:self.tiled_dim]#* pick up [B,C,224,244]


        #* from [1,1,224,244]  to [18,1,224,244], extracting the context patch  one by one 
        k=1
        for i in range(0, self.tile_factor):
            for j in range(0, 2*self.tile_factor):
                if i+j!=0:
                    tiled_targets = \
                    torch.cat((tiled_targets, targets[:, :, self.tiled_dim*i:self.tiled_dim*(i+1), self.tiled_dim*j:self.tiled_dim*(j+1)]), dim=0)
                    k += 1


        #* from [1,1,224,244]  to [18,1,224,244]
        k=1      
        for i in range(0, self.tile_factor):
            for j in range(0, 2*self.tile_factor):
                if i+j!=0:
                    tiled_inputs = \
                    torch.cat((tiled_inputs, inputs[:, :, self.tiled_dim*i:self.tiled_dim*(i+1), self.tiled_dim*j:self.tiled_dim*(j+1)]), dim=0)
                k += 1

        #* feed forward compute homography 
        _, _, distance_coeffs = self.inversenet(tiled_inputs, tiled_targets)

        
        mean_square_inverse_loss = (((distance_coeffs*distance_coeffs).sum(dim=1))**0.5).mean() 
        """"
        Square each element of the [18 * 4] matrix.
        Sum the squared elements to obtain a [18] vector.
        Take the square root of each element in the [18] vector.
        Calculate the average of the elements to obtain a [1] scalar.
        
        """
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
    
    