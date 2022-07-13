'''
Author: xushaocong
Date: 2022-06-20 21:10:45
LastEditTime: 2022-07-13 10:29:27
LastEditors: xushaocong
Description: 
FilePath: /cerberus/model/edge_model.py
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

import time
from .decoder import *
from loguru import logger

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
        self.return_intermediate_dec = True #* detr , by default  == False,  是否返回decoder 每个layer的输出, 还是只输出最后一个layer
        

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=self.return_intermediate_dec)
        #!===============================================================


        #*  sequenceial  fusion blocks 
        #?  reassemble operation 呢?  
        #? 是不是 self.scratch 的卷积部分就是对应 网络的reassemble operation ? 
        self.scratch.refinenet01 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet02 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet03 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet04 = _make_fusion_block(features, use_bn)

        #* fusion for different decoder layer 
        self.scratch.refinenet05 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet06 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet07 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet08 = _make_fusion_block(features, use_bn)

        self.scratch.refinenet09 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet10 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet11 = _make_fusion_block(features, use_bn)
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
                    #*  用refine net 将 decoder layer1 and layer 6 进行fusion得到 了size 为[160,160] 所以只需要upsample 2 倍 
                    #* refine net 已经恢复原来的size了 , 不需要进一步操作了
                    # setattr(self.scratch, "output_" + it + '_upsample', 
                    #     Interpolate(scale_factor=2, mode="bilinear", align_corners=True)
                    # )
                    
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
        #? edge_path_2 的时候不知道是不是显存不够, 跑不动!!
        decoder_out = self.decoder(decoder_input,learnable_embedding) #* (Q,KV)  ,shape == [1,WH,B,256], [ decoder_layer_number,Query number , B,inputC ]
        if self.return_intermediate_dec : 
            #* 返回的是多个decoder layer , 需要另外处理 
            #* [6,WH,B,256] == 
            decoder_out = torch.stack([ x.permute([2,3,0,1]).reshape(B,C,W,H)  for x in decoder_out.unsqueeze(1) ])
            #* pick up layer 1 and layer6
            decoder_layer1 =decoder_out[0]
            decoder_layer3 =decoder_out[2]
            decoder_layer4 =decoder_out[3]
            decoder_layer6 =decoder_out[5]
            
            #* refinenet05-09
            a= self.scratch.refinenet11(decoder_layer1) #* from [B,C,40,40] to  [B,C,80,80]
            b= self.scratch.refinenet10(decoder_layer3)#* from [B,C,40,40] to  [B,C,80,80]
            decoder_out1  = self.scratch.refinenet09(a,b)#* from [B,C,80,80] to  [B,C,160,160]
            
            c= self.scratch.refinenet08(decoder_layer4) #* from [B,C,40,40] to  [B,C,80,80]
            d= self.scratch.refinenet07(decoder_layer6)#* from [B,C,40,40] to  [B,C,80,80]
            decoder_out2  = self.scratch.refinenet06(c,d)#* from [B,C,80,80] to  [B,C,160,160]

            decoder_out = self.scratch.refinenet05(decoder_out1,decoder_out2)
            

        else:
            decoder_out =decoder_out.permute([2,3,0,1]).reshape(B,C,W,H) #* reshape back  
            

        #* rind 
        for  x in self.full_output_task_list[1:]:
            it = x[1][0]#* 每个任务只有一类 
            fun = eval("self.scratch.output_" + it)#* 全连接
            out = fun(decoder_out)
            #* refine net 已经恢复了原来的大小了, 不需要进一步操作
            # fun = eval("self.scratch.output_" + it + '_upsample')#* 上采样
            # out = fun(out)
            fun =  eval("self.scratch.output_" + it + '_sigmoid')
            out = fun(out)
            model_out.append(out)
        return model_out


      


    