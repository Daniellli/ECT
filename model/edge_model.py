'''
Author: xushaocong
Date: 2022-06-20 21:10:45
LastEditTime: 2022-06-20 22:18:14
LastEditors: xushaocong
Description: 
FilePath: /Cerberus-main/model/edge_model.py
email: xushaocong@stu.xmu.edu.cn
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel
from model.vit import forward_flex
from .blocks import (
    FeatureFusionBlock,
    FeatureFusionBlock_custom,
    Interpolate,
    _make_encoder,
    forward_vit,
)

from .decoder import *


def _make_fusion_block(features, use_bn):
    return FeatureFusionBlock_custom(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
    )


class EdgeCerberus(BaseModel):
    def __init__(
        self,
        features=256,
        backbone="vitb_rn50_384",
        readout="project",
        channels_last=False,
        use_bn=False,
        enable_attention_hooks=False,
        decoder_head_num = 8,
        
        
    ):
        super(EdgeCerberus, self).__init__()

        self.full_output_task_list = ( \
            (5, ['background']), \
            (1, ['depth']), \
            (1, ['normal']) , \
            (1, ['reflectance']),\
            (1, ['illumination'])
        )

        self.channels_last = channels_last

        hooks = {
            "vitb_rn50_384": [0, 1, 8, 11], #* 不同的backbone对应提取不同layer, 如果backbone是vitb_rn50_384 则 提取resnet 0,1 and encoder  8,11 
            "vitb16_384": [2, 5, 8, 11],    #* 如果backbone 是vitb16_384 则提取resnet  2,5 and encoder 8,11 
            "vitl16_384": [5, 11, 17, 23],  
        }

        
        #* self.pretrained : 对应backbone , 就是 resnet 50+ transformer encoder 
        #* self.scratch  对应 特征融合模块, 后面还需要接refinenet0d , 
        self.pretrained, self.scratch = _make_encoder(
            backbone,
            features,
            True,  # Set to true of you want to train from scratch, uses ImageNet weights
            groups=1,
            expand=False,
            exportable=False,
            hooks=hooks[backbone],
            use_readout=readout,
            enable_attention_hooks=enable_attention_hooks,
        )


        #* decoder 
        #* four task 
        #!===============================================================
        self.edge_query_embed = nn.Embedding(4, features)

        d_model  = features  #* 假设输入维度是这个 , detr== 256
        nhead = 8
        dim_feedforward  =2048
        dropout = 0.1 
        activation="relu" #*   detr , by default == relu, 
        normalize_before =False   #* detr , by default  == False
        num_decoder_layers= 6 #* detr == 6
        return_intermediate_dec =  False #* detr , by default  == False

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        #!===============================================================


        #*  sequenceial  fusion blocks 
        #?  reassemble operation 呢?  
        #? 是不是 self.scratch 的卷积部分就是对应 网络的reassemble operation ? 
        self.scratch.refinenet01 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet02 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet03 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet04 = _make_fusion_block(features, use_bn)

        self.scratch.refinenet05 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet06 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet07 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet08 = _make_fusion_block(features, use_bn)

        self.scratch.refinenet09 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet10 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet11 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet12 = _make_fusion_block(features, use_bn)

        self.scratch.refinenet13 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet14 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet15 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet16 = _make_fusion_block(features, use_bn)

        self.scratch.refinenet17 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet18 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet19 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet20 = _make_fusion_block(features, use_bn)

        #* final head 
        for (num_classes, output_task_list) in self.full_output_task_list:
            for it in output_task_list:
                setattr(self.scratch, "output_" + it ,nn.Sequential(
                    nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(features),
                    nn.ReLU(True),
                    nn.Dropout(0.1, False),
                    nn.Conv2d(features, num_classes, kernel_size=1),
                    # Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
                ))

                setattr(self.scratch, "output_" + it + '_upsample', 
                    Interpolate(scale_factor=2, mode="bilinear", align_corners=True)
                )
                setattr(self.scratch, "output_" + it + '_sigmoid', 
                    nn.Sigmoid()
                )
    
    '''
    description:  这个好像也没调用?
    param {*} self
    param {*} x
    param {*} name
    return {*}
    '''
    def get_attention(self, x ,name):
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)
        x = forward_flex(self.pretrained.model, x, True, name)
        return x

    '''
    description: 
    param {*} self
    param {*} x
    param {*} index : 对应当前前向传播的是哪个子任务 
    return {*}
    '''
    def forward(self, x ,index):
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x) #* 获取 resnet 1,2 and transformer encoder  9,12 layer  feature embedding 
        #* 1 . concat layer_1, layer_2, layer_3, layer_4    ,name as encoder_output
        #* 2.  s

        encoder_out = torch.cat([layer_1, layer_2, layer_3, layer_4],axis = 0 )#todo : specify the axis 

        
        decoder_out = self.decoder(encoder_out,self.edge_query_embed) #* (KV,Q)   

    

        #*  reassemble operatoion ?  将sequence 重新reassemble 成一张patch 
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        if (index == 0):
            path_4 = self.scratch.refinenet04(layer_4_rn)
            path_3 = self.scratch.refinenet03(path_4, layer_3_rn)
            path_2 = self.scratch.refinenet02(path_3, layer_2_rn)
            path_1 = self.scratch.refinenet01(path_2, layer_1_rn)
        elif (index == 1):
            path_4 = self.scratch.refinenet08(layer_4_rn)
            path_3 = self.scratch.refinenet07(path_4, layer_3_rn)
            path_2 = self.scratch.refinenet06(path_3, layer_2_rn)
            path_1 = self.scratch.refinenet05(path_2, layer_1_rn)
        elif(index == 2): #?'Module' object has no attribute 'refinenet16'
            path_4 = self.scratch.refinenet12(layer_4_rn)
            path_3 = self.scratch.refinenet11(path_4, layer_3_rn)
            path_2 = self.scratch.refinenet10(path_3, layer_2_rn)
            path_1 = self.scratch.refinenet09(path_2, layer_1_rn)
        
        elif(index==3):
            path_4 = self.scratch.refinenet16(layer_4_rn)
            path_3 = self.scratch.refinenet15(path_4, layer_3_rn)
            path_2 = self.scratch.refinenet14(path_3, layer_2_rn)
            path_1 = self.scratch.refinenet13(path_2, layer_1_rn)
        elif(index==4):
            path_4 = self.scratch.refinenet20(layer_4_rn)
            path_3 = self.scratch.refinenet19(path_4, layer_3_rn)
            path_2 = self.scratch.refinenet18(path_3, layer_2_rn)
            path_1 = self.scratch.refinenet17(path_2, layer_1_rn)
        else:
            raise RuntimeError('model index error , index must  between 0-4')
        
        output_task_list = self.full_output_task_list[index][1]
        outs = list()

        for it in output_task_list:
            fun = eval("self.scratch.output_" + it)#* 全连接
            out = fun(path_1)
            fun = eval("self.scratch.output_" + it + '_upsample')#* 上采样
            out = fun(out)
            if it != "background":
                fun =  eval("self.scratch.output_" + it + '_sigmoid')
                out = fun(out)
            outs.append(out)

        return outs



    