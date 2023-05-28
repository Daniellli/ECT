'''
Author:   "  "
Date: 2022-06-20 21:10:45
LastEditTime: 2023-03-17 13:37:59
LastEditors: daniel
Description:  concatenate the 1-st and 6-th stage decoder feature for fine-grained edge prediction
FilePath: /cerberus/model/edge_model_multi_class2.py
email:  
'''

from torchvision import transforms
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
    ResidualConvUnit_custom
)
from PIL import Image

import matplotlib.pyplot as plt
import numpy as  np 
import time
from .decoder import *
from loguru import logger
import os.path as osp
import cv2


from IPython import embed
# from sklearn.manifold import TSNE


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



class EdgeCerberusMultiClass(BaseModel):

    def __init__(
        self,
        features=256,
        backbone="vitb_rn50_384",
        readout="project",
        channels_last=False,
        use_bn=False,
        enable_attention_hooks=False,
        decoder_head_num = 8, 
        hard_edge_cls_num = 4
    ):
        super(EdgeCerberusMultiClass, self).__init__()

        self.hard_edge_cls_num = hard_edge_cls_num


    
        # self.full_output_task_list = ( \
        #     (1, ['background']), \
        #     (1, ['depth']), \
        #     (1, ['normal']) , \
        #     (1, ['reflectance']),\
        #     (1, ['illumination'])
        # )
        
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
       


        
        #*  sequenceial  fusion blocks 
        #?  reassemble operation 呢?  
        #? 是不是 self.scratch 的卷积部分就是对应 网络的reassemble operation ? 
        self.scratch.refinenet01 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet02 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet03 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet04 = _make_fusion_block(features, use_bn)#* output [160,160]
        

        #* 下采样作为decoder 的输入 
        #? 这里是否要替换成  卷积操作? 
        setattr(self.scratch, "output_downsample",
            Interpolate(scale_factor=0.25, mode="bilinear", align_corners=True))#* from [160,160] to [40,40]



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
        
        self.return_attention = False
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before,
                                                return_attention = self.return_attention)



        decoder_norm = nn.LayerNorm(d_model)

        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=self.return_intermediate_dec,
                                          return_attention = self.return_attention
                                          )

        #* fusion for different decoder layer 
        #*  pick 2 layer  to upsampling to [160,160 ] by refine net 
        self.scratch.refinenet09 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet10 = _make_fusion_block(features, use_bn) 
        self.scratch.refinenet11 = _make_fusion_block(features, use_bn)


        self.final_dropout1 = nn.Dropout(dropout)
        concated_channel = d_model*2
        self.final_norm1 = nn.BatchNorm2d(concated_channel)
        self.final_rcu = ResidualConvUnit_custom(concated_channel,_get_activation_fn(activation),True)
   

        #* final head 
        #* conv and ConvTranspose2d to [320,320] and classify
        for cls_idx in range(self.hard_edge_cls_num+1):
            num_classes = 1

            if cls_idx==0:
                setattr(self.scratch, "output_" + str(cls_idx) ,nn.Sequential(
                    nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(features),
                    nn.ReLU(True),
                    nn.Dropout(0.1, False),
                    nn.Conv2d(features, num_classes, kernel_size=1),
                ))
                
            else:
                setattr(self.scratch, "output_" + str(cls_idx) ,nn.Sequential(
                    nn.Conv2d(concated_channel, concated_channel, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(concated_channel),
                    nn.ReLU(True),
                    nn.Dropout(0.1, False),
                    nn.Conv2d(concated_channel, num_classes, kernel_size=1),
                ))

            #* 需要 batch normalization 吗? 
            setattr(self.scratch, "output_" + str(cls_idx) + '_upsample', 
                nn.Sequential(
                # Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
                nn.ConvTranspose2d(num_classes, num_classes, kernel_size=2, stride=2, bias=False),#* 比interpolate 多了100个参数
                nn.BatchNorm2d(num_classes),
                # nn.ReLU(inplace=True)
                nn.Sigmoid()
                )
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

        x = forward_flex(self.pretrained.model, x, True, name) #* true mean plot attention,   
        return x

    '''
    description: 
    param {*} self
    param {*} x
    param {*} index : 对应当前前向传播的是哪个子任务 
    return {*}
    '''
    def forward(self, x ):
        if self.return_attention:
            origin_image= x.clone()
        _B,_C,_H,_W=x.shape
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

        out = self.scratch.output_0(edge_path_1)
        model_out.append(self.scratch.output_0_upsample(out))
        

        #*==============================
        decoder_input=self.scratch.output_downsample(edge_path_1)#* downsample from  (B,256,160,160) to   (B,256,40,40)
        #*==============================


        B,C,W,H=decoder_input.shape
        decoder_input=  decoder_input.permute([2,3,0,1]).reshape([-1,B,C])  #*(B,C,W,H)  to (WH, B,C)
        
        learnable_embedding = self.edge_query_embed.weight.unsqueeze(1).repeat(1,B,1)#* (query_num,C) --> (query_num,B,C)

        #? edge_path_2 的时候不知道是不是显存不够, 跑不动!!
        if self.return_attention:
            unloader = transforms.ToPILImage()
            # unloader(origin_image.cpu().clone().squeeze(0)).save(f'origin.jpg')
            cv2.imwrite('origin.jpg',origin_image.cpu().clone().squeeze().permute(1,2,0).numpy()*255)


            decoder_out,attentions = self.decoder(decoder_input,learnable_embedding) #* (Q,KV)  ,shape == [1,WH,B,256], [ decoder_layer_number,Query number , B,inputC ]
            #todo vis attentions
            # attentions = torch.stack([ x.permute([0,3,2,1]).reshape(B,4,W,H)  for x in attentions.unsqueeze(1) ])
            attentions = torch.stack([ x.permute([0,3,1,2]).reshape(B,4,W,H)  for x in attentions.unsqueeze(1) ])

            #* traverse 6 decoder layer 
            
            #*=================
            with open('tmp.txt','r') as f:
                atten_path = f.readlines()
            #*=================
            
            for iidx,atten in enumerate(attentions):
                #* traverse 4 attention map 
                for  idx, (attention,x) in enumerate(zip(atten.squeeze(),self.full_output_task_list[1:])):
                    name = x[1][0]
                    #* 乘不乘255 都一样
                    attention_map = F.interpolate(attention.unsqueeze(0).unsqueeze(0),scale_factor=8,mode='bilinear')#*  attention == [40,60] ->[320,480]
                    attention_map = attention_map.cpu().clone().squeeze().numpy()

                    #* plan 1 
                    # plt.imsave(fname=osp.join(atten_path[0],f'atten-{name}-{iidx}.jpg'), arr=attention_map, format='png')
                    plt.imsave(fname=osp.join(atten_path[0],f'atten-{name}-{iidx}.jpg'), arr=attention_map, format='png',cmap='jet')
                    #* plan 2
                    # figure = plt.figure()
                    # plt.pcolor(test, cmap='jet')
                    # # plt.colorbar()
                    # # plt.savefig('tmp.jpg')
                    # plt.xticks([])
                    # plt.yticks([])
                    # plt.axis('off')
                    
        else :
            decoder_out = self.decoder(decoder_input,learnable_embedding) #* (Q,KV)  ,shape == [1,WH,B,256], [ decoder_layer_number,Query number , B,inputC ]



        if self.return_intermediate_dec : 
            #* 返回的是多个decoder layer , 需要另外处理 
            #* [6,WH,B,256] == 
            decoder_out = torch.stack([ x.permute([2,3,0,1]).reshape(B,C,W,H)  for x in decoder_out.unsqueeze(1) ])
            #* pick up layer 1 and layer6
            decoder_layer1 =decoder_out[0]
            decoder_layer6 =decoder_out[-1]
            #* refinenet05-09
            a= self.scratch.refinenet11(decoder_layer1) #* from [B,C,40,40] to  [B,C,80,80]
            b= self.scratch.refinenet10(decoder_layer6)#* from [B,C,40,40] to  [B,C,80,80]
            decoder_out  = self.scratch.refinenet09(a,b)#* from [B,C,80,80] to  [B,C,160,160]
        else:
            decoder_out =decoder_out.permute([2,3,0,1]).reshape(B,C,W,H) #* reshape back  
            

        #* decoder_out 正则化  , 
        # !+===========================        
        # decoder_out  = edge_path_1 + self.final_dropout1(decoder_out)
            
        # embed()
        # b = self.final_rcu(self.final_norm1torch.cat([edge_path_1, self.final_dropout1(decoder_out)],dim=1)))
        # c = self.scratch.output_1(b)

        decoder_out  = torch.cat([edge_path_1, self.final_dropout1(decoder_out)],dim=1)
        decoder_out = self.final_norm1(decoder_out)
        decoder_out = self.final_rcu(decoder_out)

        
        # !+===========================
        

        #* rind 
        for cls_idx in range(self.hard_edge_cls_num):
            fun = eval("self.scratch.output_" + str(cls_idx+1))#* 全连接
            out = fun(decoder_out)

            #* refine net 已经恢复了原来的大小了, 不需要进一步操作
            fun = eval("self.scratch.output_" + str( cls_idx+1) + '_upsample')#* 上采样
            out = fun(out)
            model_out.append(out)

            
        return model_out


      


'''
description: 传过来的都是PIL 的image类型 
param {*} gray_img interpolate输出的灰度图
param {*} origin_img
param {*} save_name
return {*}
'''
def blend_atten_origin_image(gray_img,origin_img,save_name):
    figure = plt.figure()
    plt.pcolor(gray_img, cmap='jet')
    # plt.colorbar()
    # plt.savefig('tmp.jpg')
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    #* 将裁减的数据和原来数据融合到一起
    To_tensor=transforms.ToTensor()

    a =To_tensor(fig2data(figure).convert('RGB'))
    center = np.array(a.shape[-2:])/2


    #* crop a 
    # todo      map (480,640) 和origin image (480,320)大小 不一致, 一个 
    # after_blend=Image.blend(,origin_img.convert('RGBA'),0.8)

    # d =  cv2.cvtColor(np.asarray(after_blend),cv2.COLOR_RGB2BGR) 

    # cv2.imwrite(save_name,d)
    

'''
description:  可视化attention
param {*} atten: (H,W) map 
param {*} save_name
return {*}
'''
# def down_dim_and_vis_atten(atten,save_name):
#     H,W=atten.shape
#     tsne = TSNE(n_components=1, init='pca', random_state=0)
#     # temp = atten.view(H*W).cpu().clone().unsqueeze(0)
#     temp = atten.view(H*W,1).cpu().clone()
#     #temp = q_feat.squeeze(0).cpu()
#     result = tsne.fit_transform(temp)
#     result = result.reshape([H,W])

#     result = result - result.min()
#     result = result/result.max()

#     plt.imsave(fname=save_name, arr=result, format='png',cmap='plasma')
    

'''
description:  将matplotlib 数据 转image
param {*} self
param {*} fig
return {*}
'''
def fig2data(fig):
    """
    fig = plt.figure()
    image = fig2data(fig)
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    import PIL.Image as Image
    # draw the renderer
    fig.canvas.draw()
    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    image = Image.frombytes("RGBA", (w, h), buf.tostring())
    # image = np.asarray(image)
    return image