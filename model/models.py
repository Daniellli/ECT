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


def _make_fusion_block(features, use_bn):
    return FeatureFusionBlock_custom(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
    )


class DPT(BaseModel):
    def __init__(
        self,
        head,
        features=256,
        backbone="vitb_rn50_384",
        readout="project",
        channels_last=False,
        use_bn=False,
        enable_attention_hooks=False,
    ):

        super(DPT, self).__init__()

        self.channels_last = channels_last

        hooks = {
            "vitb_rn50_384": [0, 1, 8, 11],
            "vitb16_384": [2, 5, 8, 11],
            "vitl16_384": [5, 11, 17, 23],
        }

        # Instantiate backbone and reassemble blocks
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

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.scratch.output_conv = head

    def forward(self, x):
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = path_1
        for i, it in enumerate(self.scratch.output_conv):
            out = it(out)
            if i == 4:
                feature = out.clone()
        return [out], [feature]


class Cerberus(BaseModel):
    def __init__(
        self,
        head,
        features=256,
        backbone="vitb_rn50_384",
        readout="project",
        channels_last=False,
        use_bn=False,
        enable_attention_hooks=False,
    ):

        super(Cerberus, self).__init__()

        self.channels_last = channels_last

        hooks = {
            "vitb_rn50_384": [0, 1, 8, 11], 
            "vitb16_384": [2, 5, 8, 11],    
            "vitl16_384": [5, 11, 17, 23],  
        }

        
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

        #* 特征融合模块? 
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

        #!+================ 
        self.scratch.refinenet13 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet14 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet15 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet16 = _make_fusion_block(features, use_bn)

        self.scratch.refinenet17 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet18 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet19 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet20 = _make_fusion_block(features, use_bn)
        #!+================





class DPTDepthModel(DPT):
    def __init__(self, path=None, non_negative=True, **kwargs):
        features = kwargs["features"] if "features" in kwargs else 256

        head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
            nn.Identity(),
        )

        super().__init__(head, **kwargs)

        if path is not None:
            self.load(path)

    def forward(self, x):
        return super().forward(x).squeeze(dim=1)

class DPTSegmentationModel(DPT):
    def __init__(self, num_classes, path=None, **kwargs):

        features = kwargs["features"] if "features" in kwargs else 256

        kwargs["use_bn"] = True

        head = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            nn.Dropout(0.1, False),
            nn.Conv2d(features, num_classes, kernel_size=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
        )

        super().__init__(head, **kwargs)

        self.auxlayer = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            nn.Dropout(0.1, False),
            nn.Conv2d(features, num_classes, kernel_size=1),
        )

        if path is not None:
            self.load(path)

class DPTSegmentationModelMultiHead(DPT):
    def __init__(self, num_classes, output_task_list, path=None, **kwargs):
        self.output_task_list = output_task_list

        features = kwargs["features"] if "features" in kwargs else 256

        kwargs["use_bn"] = True

        head = None

        super().__init__(head, **kwargs)
        
        for it in output_task_list:
            setattr(self.scratch, "output_" + it ,nn.Sequential(
                nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(features),
                nn.ReLU(True),
                nn.Dropout(0.1, False),
                nn.Conv2d(features, num_classes, kernel_size=1),
                Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            ))

        if path is not None:
            self.load(path)

    def forward(self, x):
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        outs = list()
        features = list()

        for it in self.output_task_list:
            fun = eval("self.scratch.output_" + it)
            out = path_1
            for j, jt in enumerate(fun):
                out = jt(out)
                if j == 4:
                    feature = out.clone()
            outs.append(out)
            features.append(feature)
        return outs, features

class TransferNet(nn.Module):
    def __init__(self,
        input_task_list,
        output_task_list,
        **kwargs):

        features = kwargs["features"] if "features" in kwargs else 256

        self.input_task_list = input_task_list
        self.output_task_list = output_task_list
        
        

        if len(input_task_list) == 1:
            self.in_channels = 40
        else:
            self.in_channels = len(input_task_list) * 2

        if len(output_task_list) == 1:
            self.num_classes = 40
        else:
            self.num_classes = 2

        super(TransferNet, self).__init__()

        self.transfer = nn.Sequential(
            nn.Conv2d(self.in_channels, features, kernel_size=3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
        )

        for it in output_task_list:
            setattr(self, "output_" + it, nn.Sequential(
                nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(features),
                nn.ReLU(True),
                nn.Dropout(0.1, False),
                nn.Conv2d(features, self.num_classes, kernel_size=1),
                Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            ))
        


    def forward(self,x):
        if (len(self.input_task_list) > 1):
            cat_x = torch.cat(tuple(x), 1)
        else:
            [cat_x] = x
        transfer_x = self.transfer(cat_x)

        outs = list()

        for it in self.output_task_list:
            fun = eval("self.output_" + it)
            out = fun(transfer_x)
            outs.append(out)
        
        return outs


class CerberusSegmentationModelMultiHead(Cerberus):
    def __init__(self, path=None, **kwargs):

        features = kwargs["features"] if "features" in kwargs else 256

        kwargs["use_bn"] = True

        head = None

        super().__init__(head, **kwargs)
        #!======================
        # full_output_task_list = ( \
        #     (2, ['Wood','Painted','Paper','Glass','Brick','Metal','Flat','Plastic','Textured','Glossy','Shiny']), \
        #     (2, ['L','M','R','S','W']), \
        #     (40, ['Segmentation']) \
        # )
        full_output_task_list = ( \
            (5, ['background']), \
            (1, ['depth']), \
            (1, ['normal']) , \
            (1, ['reflectance']),\
            (1, ['illumination'])
        )

        self.full_output_task_list = full_output_task_list

   
        
        for (num_classes, output_task_list) in full_output_task_list:
            for it in output_task_list:
                setattr(self.scratch, "output_" + it ,nn.Sequential(
                    nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(features),
                    #!+=========
                    nn.ReLU(True),
                    #!+=========
                    nn.Dropout(0.1, False),
                    nn.Conv2d(features, num_classes, kernel_size=1),
                    # Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
                ))

                setattr(self.scratch, "output_" + it + '_upsample', 
                    Interpolate(scale_factor=2, mode="bilinear", align_corners=True)
                )

                #!=================== sigmoid 输出 每个像素点是边缘的概率
                setattr(self.scratch, "output_" + it + '_sigmoid', 
                    nn.Sigmoid()
                )
                #!===================
            

        if path is not None:
            self.load(path)
        else:
            pass

    def get_attention(self, x ,name):
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        x = forward_flex(self.pretrained.model, x, True, name)

        return x

    def forward(self, x ,index):
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)
        #*( B,256,80,80), ( B,512,40,40),( B,768,20,20),( B,768,10,10)
        layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x) 

        #*  reassemble operatoion ?   
        layer_1_rn = self.scratch.layer1_rn(layer_1) #* ( B,256,80,80)
        layer_2_rn = self.scratch.layer2_rn(layer_2)#* ( B,256,40,40)
        layer_3_rn = self.scratch.layer3_rn(layer_3)#* ( B,256,20,20)
        layer_4_rn = self.scratch.layer4_rn(layer_4)#* ( B,256,10,10)

        
        if (index == 0):
            path_4 = self.scratch.refinenet04(layer_4_rn)#* ( B,256,20,20)
            path_3 = self.scratch.refinenet03(path_4, layer_3_rn)#* ( B,256,40,40)
            path_2 = self.scratch.refinenet02(path_3, layer_2_rn)#* ( B,256,80,80)
            path_1 = self.scratch.refinenet01(path_2, layer_1_rn)#* ( B,256,160,160)
        elif (index == 1):
            path_4 = self.scratch.refinenet08(layer_4_rn)
            path_3 = self.scratch.refinenet07(path_4, layer_3_rn)
            path_2 = self.scratch.refinenet06(path_3, layer_2_rn)
            path_1 = self.scratch.refinenet05(path_2, layer_1_rn)
        elif(index == 2):#?'Module' object has no attribute 'refinenet16'
            path_4 = self.scratch.refinenet12(layer_4_rn)
            path_3 = self.scratch.refinenet11(path_4, layer_3_rn)
            path_2 = self.scratch.refinenet10(path_3, layer_2_rn)
            path_1 = self.scratch.refinenet09(path_2, layer_1_rn)
        #!========================
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
        #!=======================
        else:
            assert 0 == 1
        

        
        output_task_list = self.full_output_task_list[index][1]
        
        outs = list()
        
        for it in output_task_list:
            fun = eval("self.scratch.output_" + it)#*
            out = fun(path_1)
            fun = eval("self.scratch.output_" + it + '_upsample')#* 
            out = fun(out)
            #!==========
            if it != "background":
                fun =  eval("self.scratch.output_" + it + '_sigmoid')
                out = fun(out)
            #!==========
            outs.append(out)
        #!=============
        # return outs,  [self.sigma.sub_attribute_sigmas, 
        #             self.sigma.sub_affordance_sigmas,
        #             self.sigma.sub_seg_sigmas, 
        #             self.sigma.attribute_sigmas, 
        #             self.sigma.affordance_sigmas, 
        #             self.sigma.seg_sigmas], []

        return outs,  None, None


        #!=============


#################### two head ##################

class TwoHeadDPT(BaseModel):
    def __init__(
        self,
        head,
        features=256,
        backbone="vitb_rn50_384",
        readout="project",
        channels_last=False,
        use_bn=False,
        enable_attention_hooks=False,
    ):

        super(TwoHeadDPT, self).__init__()

        self.channels_last = channels_last

        hooks = {
            "vitb_rn50_384": [0, 1, 8, 11],
            "vitb16_384": [2, 5, 8, 11],
            "vitl16_384": [5, 11, 17, 23],
        }

        # Instantiate backbone and reassemble blocks
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

        self.scratch.refinenet01 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet02 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet03 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet04 = _make_fusion_block(features, use_bn)

        self.scratch.refinenet05 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet06 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet07 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet08 = _make_fusion_block(features, use_bn)


class TwoHeadDPTSegmentationModel(TwoHeadDPT):
    def __init__(self, path=None, **kwargs):

        features = kwargs["features"] if "features" in kwargs else 256

        kwargs["use_bn"] = True

        head = None

        super().__init__(head, **kwargs)

        full_output_task_list = ( \
            (2, ['Wood','Painted','Paper','Glass','Brick','Metal','Flat','Plastic','Textured','Glossy','Shiny']), \
            (2, ['L','M','R','S','W']), \
            (40, ['Segmentation']) \
        )

        self.full_output_task_list = full_output_task_list
        self.add_module('sigma',nn.Module())

        # self.sigma.attribute_sigmas = nn.Parameter(torch.Tensor(1).uniform_(-1.60, 0.0), requires_grad=True)
        # self.sigma.affordance_sigmas = nn.Parameter(torch.Tensor(1).uniform_(-1.60, 0.0), requires_grad=True)
        # self.sigma.seg_sigmas = nn.Parameter(torch.Tensor(1).uniform_(-1.60, 0.0), requires_grad=True)

        # elf.sigma.sub_attribute_sigmas = nn.Parameter(torch.Tensor(len(full_output_task_list[0][1])).uniform_(-1.60, 0.0), requires_grad=True)
        # self.sigma.sub_affordance_sigmas = nn.Parameter(torch.Tensor(len(full_output_task_list[1][1])).uniform_(-1.60, 0.0), requires_grad=True)
        # self.sigma.sub_seg_sigmas = nn.Parameter(torch.Tensor(len(full_output_task_list[2][1])).uniform_(-1.60, 0.0), requires_grad=True)
        
        # self.sigma.attribute_sigmas = nn.Parameter(torch.Tensor(1).uniform_(0.20, 1.0), requires_grad=True)
        # self.sigma.affordance_sigmas = nn.Parameter(torch.Tensor(1).uniform_(0.20, 1.0), requires_grad=True)
        # self.sigma.seg_sigmas = nn.Parameter(torch.Tensor(1).uniform_(0.20, 1.0), requires_grad=True)

        # self.sigma.sub_attribute_sigmas = nn.Parameter(torch.Tensor(len(full_output_task_list[0][1])).uniform_(0.20, 1.0), requires_grad=True)
        # self.sigma.sub_affordance_sigmas = nn.Parameter(torch.Tensor(len(full_output_task_list[1][1])).uniform_(0.20, 1.0), requires_grad=True)
        # self.sigma.sub_seg_sigmas = nn.Parameter(torch.Tensor(len(full_output_task_list[2][1])).uniform_(0.20, 1.0), requires_grad=True)
        


        for (num_classes, output_task_list) in full_output_task_list:
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
            

        if path is not None:
            self.load(path)
        else:
            pass

    def forward(self, x ,index):
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x)


        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        if (index == 0):
            path_4 = self.scratch.refinenet04(layer_4_rn)
            path_3 = self.scratch.refinenet03(path_4, layer_3_rn)
            path_2 = self.scratch.refinenet02(path_3, layer_2_rn)
            path_1 = self.scratch.refinenet01(path_2, layer_1_rn)
        elif (index == 2):
            path_4 = self.scratch.refinenet08(layer_4_rn)
            path_3 = self.scratch.refinenet07(path_4, layer_3_rn)
            path_2 = self.scratch.refinenet06(path_3, layer_2_rn)
            path_1 = self.scratch.refinenet05(path_2, layer_1_rn)
        else:
            assert 0 == 1
        
        output_task_list = self.full_output_task_list[index][1]

        outs = list()

        for it in output_task_list:
            fun = eval("self.scratch.output_" + it)
            out = fun(path_1)
            fun = eval("self.scratch.output_" + it + '_upsample')
            out = fun(out)
            outs.append(out)

        return outs,  [#self.sigma.sub_attribute_sigmas, 
                    # self.sigma.sub_affordance_sigmas,
                    # self.sigma.sub_seg_sigmas, 
                    # #self.sigma.attribute_sigmas, 
                    # self.sigma.affordance_sigmas, 
                    # self.sigma.seg_sigmas
                    ], []

