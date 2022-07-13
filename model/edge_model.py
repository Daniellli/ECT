'''
Author: xushaocong
Date: 2022-06-20 21:10:45
LastEditTime: 2022-07-13 19:33:00
LastEditors: xushaocong
Description: 
FilePath: /cerberus/model/edge_model.py
email: xushaocong@stu.xmu.edu.cn
'''


from turtle import forward
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

import time
from .decoder import *
from loguru import logger

from IPython import embed
def _make_fusion_block(features, use_bn):
    return FeatureFusionBlock_custom(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
    )





def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")



'''
description:  before decoder, embed the interaction between the different learnable embedding 
return {*}
'''
class QueryAttention(nn.Module):
    

    def __init__(self,d_model  = 256 ,nhead = 8 ,dim_feedforward  =2048,dropout = 0.1 ,
                    activation="relu") -> None:

        super(QueryAttention, self).__init__()


        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout) #
        #* without position embedding 
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)#
        self.dropout = nn.Dropout(dropout)#
        self.linear2 = nn.Linear(dim_feedforward, d_model)#
        self.norm1 = nn.LayerNorm(d_model)#
        self.norm2 = nn.LayerNorm(d_model)#
        self.dropout1 = nn.Dropout(dropout)#
        self.dropout2 = nn.Dropout(dropout)#
        self.activation = _get_activation_fn(activation)#

        
        
    '''
    description:  
    param {*} self
    param {*} src:single  query embedding  ,  
    return {*}
    '''
    def forward(self,src ):
        src2 = self.self_attn(src, src, value=src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src



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
        #!===============================================================
        input_dim =  features #* 256 
        self.edge_query_embed = nn.Embedding(4, input_dim)
        d_model  = input_dim 
        nhead = 8 #* detr ==8
        dim_feedforward  =2048
        dropout = 0.1 
        activation="relu" #*   detr , by default == relu, 
        normalize_before =False   #* detr , by default  == False
        num_decoder_layers= 6 #* detr == 6
        return_intermediate_dec =  False #* detr , by default  == False,  是否返回decoder 每个layer的输出, 还是只输出最后一个layer

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)
        #!===============================================================

        #! before decoder, embed the interaction between the different learnable embedding 
        #!===============================================================  
        # self.quer1=QueryAttention(nhead=8)
        # self.quer2=QueryAttention(nhead=8)
        # self.quer3=QueryAttention(nhead=8)
        # self.quer4=QueryAttention(nhead=8)

        # self.quer1=QueryAttention(nhead=1)
        # self.quer2=QueryAttention(nhead=1)
        # self.quer3=QueryAttention(nhead=1)
        # self.quer4=QueryAttention(nhead=1)
        #!=============================================================== 

        #*  sequenceial  fusion blocks 
        #?  reassemble operation 呢?  
        #? 是不是 self.scratch 的卷积部分就是对应 网络的reassemble operation ? 
        self.scratch.refinenet01 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet02 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet03 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet04 = _make_fusion_block(features, use_bn)

        

        # self.scratch.refinenet05 = _make_fusion_block(features, use_bn)
        # self.scratch.refinenet06 = _make_fusion_block(features, use_bn)
        # self.scratch.refinenet07 = _make_fusion_block(features, use_bn)
        # self.scratch.refinenet08 = _make_fusion_block(features, use_bn)

        # self.scratch.refinenet09 = _make_fusion_block(features, use_bn)
        # self.scratch.refinenet10 = _make_fusion_block(features, use_bn)
        # self.scratch.refinenet11 = _make_fusion_block(features, use_bn)
        # self.scratch.refinenet12 = _make_fusion_block(features, use_bn)

        # self.scratch.refinenet13 = _make_fusion_block(features, use_bn)
        # self.scratch.refinenet14 = _make_fusion_block(features, use_bn)
        # self.scratch.refinenet15 = _make_fusion_block(features, use_bn)
        # self.scratch.refinenet16 = _make_fusion_block(features, use_bn)

        # self.scratch.refinenet17 = _make_fusion_block(features, use_bn)
        # self.scratch.refinenet18 = _make_fusion_block(features, use_bn)
        # self.scratch.refinenet19 = _make_fusion_block(features, use_bn)
        # self.scratch.refinenet20 = _make_fusion_block(features, use_bn)

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

                if it == "background":
                    setattr(self.scratch, "output_" + it + '_upsample', 
                        Interpolate(scale_factor=2, mode="bilinear", align_corners=True)
                    )
                else :
                    setattr(self.scratch, "output_" + it + '_upsample', 
                        Interpolate(scale_factor=8, mode="bilinear", align_corners=True)
                    )
                    
                    setattr(self.scratch, "output_" + it + '_sigmoid', 
                        nn.Sigmoid()
                    )

        setattr(self.scratch, "output_downsample",
         Interpolate(scale_factor=0.25, mode="bilinear", align_corners=True))


    
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
    def forward(self, x ):
        B,C,H,W=x.shape
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)
        #* layer1 : (B,256,80,80)
        #* layer2 : (B,512,40,40)
        #* layer3 : (B,768,20,20)
        #* layer4 : (B,768,10,10)
        layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x) #* 获取 resnet 1,2 and transformer encoder  9,12 layer  feature embedding 
        
        #*  reassemble operatoion ?  将sequence 重新reassemble 成一张patch 
        layer_1_rn = self.scratch.layer1_rn(layer_1)#*  : (B,256,80,80)
        layer_2_rn = self.scratch.layer2_rn(layer_2)#*  : (B,256,40,40)
        layer_3_rn = self.scratch.layer3_rn(layer_3)#*  : (B,256,20,20)
        layer_4_rn = self.scratch.layer4_rn(layer_4)#*  : (B,256,10,10)

        edge_path_4 = self.scratch.refinenet04(layer_4_rn)#* enlarge to  (B,256,20,20)
        edge_path_3 = self.scratch.refinenet03(edge_path_4, layer_3_rn)#* fusion to  (B,256,40,40) ,  spend  0s  to pass decoder 
        edge_path_2 = self.scratch.refinenet02(edge_path_3, layer_2_rn)#* fusion to  (B,256,80,80), 
        edge_path_1 = self.scratch.refinenet01(edge_path_2, layer_1_rn)#* fusion to  (B,256,160,160)


        model_out = []
        #* background 
        edge_it = self.full_output_task_list[0][1][0]
        fun = eval("self.scratch.output_" + edge_it)#* 全连接
        out = fun(edge_path_1)
        fun = eval("self.scratch.output_" + edge_it + '_upsample')#* 上采样
        model_out.append(fun(out))
        

        #*==============================
        decoder_input=self.scratch.output_downsample(edge_path_1)#* downsample from  (B,256,160,160) to   (B,256,40,40)
        #*==============================


        B,C,W,H=decoder_input.shape
        decoder_input=  decoder_input.permute([2,3,0,1]).reshape([-1,B,C])  #*(B,C,W,H)  to (WH, B,C)
        
        learnable_embedding = self.edge_query_embed.weight.unsqueeze(1).repeat(1,B,1)#* (query_num,C) --> (query_num,B,C)
        #*================================= add interaction between the   different queries 
        # depth_query = self.quer1(learnable_embedding[0].unsqueeze(0))
        # normal_query = self.quer2(learnable_embedding[1].unsqueeze(0))
        # reflectance_query = self.quer3(learnable_embedding[2].unsqueeze(0))
        # illumination_query = self.quer4(learnable_embedding[3].unsqueeze(0))
        # learnable_embedding = torch.cat([depth_query,normal_query,reflectance_query,illumination_query])
        #*=================================

        #? edge_path_2 的时候不知道是不是显存不够, 跑不动!!
        decoder_out = self.decoder(decoder_input,learnable_embedding) #* (Q,KV)  ,shape == [1,WH,B,256], [ decoder_layer_number,Query number , B,inputC ]



        decoder_out =decoder_out.permute([2,3,0,1]).reshape(B,C,W,H) #* reshape back  

        #* rind 
        for  x in self.full_output_task_list[1:]:
            it = x[1][0]#* 每个任务只有一类 
            fun = eval("self.scratch.output_" + it)#* 全连接
            out = fun(decoder_out)
            fun = eval("self.scratch.output_" + it + '_upsample')#* 上采样
            out = fun(out)
            fun =  eval("self.scratch.output_" + it + '_sigmoid')
            out = fun(out)
            model_out.append(out)
        return model_out


      


    