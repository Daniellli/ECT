'''
Author:   "  "
Date: 2022-06-20 22:49:32
LastEditTime: 2023-08-06 22:38:34
LastEditors: daniel
Description: 
FilePath: /Cerberus-main/transferibility/test_IIW.py
email:  
'''


import cv2

import os
# from IPython import embed #for terminal debug 
# import pdb
import time
import numpy as np
import sys
import torch
import torch.backends.cudnn as cudnn
from torchvision import  transforms
from torch.autograd import Variable

import data_transforms as transforms
from model.models import  CerberusSegmentationModelMultiHead
from model.edge_model import  EdgeCerberus
import torch 
import os.path as osp
from tqdm import tqdm
import scipy.io as sio
import torchvision.transforms as transforms

#*===================
import glob
import wandb
from loguru import logger
from utils.loss import SegmentationLosses
from utils.edge_loss2 import AttentionLoss2
from dataloaders.datasets.bsds_hd5 import Mydataset
from dataloaders.datasets.nyud_geonet import NYUD_GeoNet
from dataloaders.datasets.sbu import SBU
from dataloaders.datasets.istd import ISTD
from dataloaders.datasets.iiw_dataset import IIWDataset

from torch.utils.data.distributed import DistributedSampler
from utils import make_dir,parse_args
import json
import warnings
warnings.filterwarnings('ignore')



from utils.global_var import *









'''
description: 
param {*} args
return {*}
'''
def test_edge(model_abs_path,test_loader,save_name,runid=None,):
    tic = time.time()
    
    a = osp.split(model_abs_path)
    if runid is  None:
        output_dir  = osp.join(a[0],"..",save_name)
    else:
        output_dir  = osp.join(a[0],"..","%s_%d"%(save_name,runid))



    
    # output_dir = osp.join("/".join(args.resume.split("/")[:-2]),"model_res")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # logger.info(output_dir)
    
    
    
    # single_model = EdgeCerberus(backbone="vitb_rn50_384")
    # single_model = EdgeCerberus(backbone="vitb_rn50_384",enable_attention_hooks=True)
    single_model = EdgeCerberus(backbone="vitb_rn50_384")

    
    checkpoint = torch.load(model_abs_path,map_location='cuda:0')
    for name, param in checkpoint['state_dict'].items():
        name = name.replace("module.","") 
        single_model.state_dict()[name].copy_(param)

    logger.info("load model done ")
    model = single_model.cuda()
    model.eval()

    cudnn.benchmark = True
    edge_output_dir = os.path.join(output_dir, 'all_edges/met')
    make_dir(edge_output_dir)

    depth_output_dir = os.path.join(output_dir, 'depth/met')
    make_dir(depth_output_dir)
    
    normal_output_dir = os.path.join(output_dir, 'normal/met')
    make_dir(normal_output_dir)

    reflectance_output_dir = os.path.join(output_dir, 'reflectance/met')
    make_dir(reflectance_output_dir)

    illumination_output_dir = os.path.join(output_dir, 'illumination/met')
    make_dir(illumination_output_dir)


    # attention_output_dir = os.path.join(output_dir, 'attention')
    # make_dir(attention_output_dir)
    
    logger.info("dir prepare done ,start to reference  ")
    if not(len(glob.glob(normal_output_dir+"/*.mat")) == len(test_loader)): 
        model.eval()
        tbar = tqdm(test_loader, desc='\r')
        for i, (image,name) in enumerate(tbar):#*  B,C,H,W_OK

            
            name= name[0]
            # logger.info(name)
            image = Variable(image, requires_grad=False)
            image = image.cuda()
            
            B,C,H,W = image.shape 
            # trans1 = transforms.Compose([transforms.Resize(size=(H//16*16, W//16*16))])
            trans1 = transforms.Compose([transforms.Resize(size=(H//32*32, W//32*32))])
            trans2 = transforms.Compose([transforms.Resize(size=(H, W))])
            image = trans1(image)

            
            with torch.no_grad():
                #!======================
                # model.get_attention(image,attention_save_dir) 
                # vis_att(model,image)
                #!======================
                res= model(image)#* out_background,out_depth, out_normal, out_reflectance, out_illumination

            out_edge = trans2(res[0])
            out_depth, out_normal, out_reflectance, out_illumination = trans2(res[1]),trans2(res[2]),trans2(res[3]),trans2(res[4])


            edge_pred = out_edge.data.cpu().numpy()
            edge_pred = edge_pred.squeeze()
            sio.savemat(os.path.join(edge_output_dir, '{}.mat'.format(name)), {'result': edge_pred})
            
            depth_pred = out_depth.data.cpu().numpy()
            depth_pred = depth_pred.squeeze()
            sio.savemat(os.path.join(depth_output_dir, '{}.mat'.format(name)), {'result': depth_pred})


            normal_pred = out_normal.data.cpu().numpy()
            normal_pred = normal_pred.squeeze()
            sio.savemat(os.path.join(normal_output_dir, '{}.mat'.format(name)), {'result': normal_pred})

            reflectance_pred = out_reflectance.data.cpu().numpy()
            reflectance_pred = reflectance_pred.squeeze()
            sio.savemat(os.path.join(reflectance_output_dir, '{}.mat'.format(name)), {'result': reflectance_pred})

            illumination_pred = out_illumination.data.cpu().numpy()
            illumination_pred = illumination_pred.squeeze()
            sio.savemat(os.path.join(illumination_output_dir, '{}.mat'.format(name)),
                        {'result': illumination_pred})
        

    #* just for attention 
    logger.info("reference done , start to eval ")
    
    eval_res= None
    if False:
        #* Due to environment conflicts, activate another virtual environment in a separate shell to perform the evaluation.
        #! Set the second parameter to 1 to test edges.
        os.system("./eval_tools/test.sh %s %s"%(output_dir,"1"))
        logger.info("eval done  ")
        with open (osp.join(output_dir,"eval_res.json"),'r')as f :
            eval_res = json.load(f)

    spend_time =  time.time() - tic
    logger.info("spend time : "+time.strftime("%H:%M:%S",time.gmtime(spend_time)))
    return eval_res


def edge_validation(model,test_loader,output_dir):    

    
    edge_output_dir = os.path.join(output_dir, 'edge/met')
    make_dir(edge_output_dir)

    depth_output_dir = os.path.join(output_dir, 'depth/met')
    make_dir(depth_output_dir)
    
    normal_output_dir = os.path.join(output_dir, 'normal/met')
    make_dir(normal_output_dir)

    reflectance_output_dir = os.path.join(output_dir, 'reflectance/met')
    make_dir(reflectance_output_dir)

    illumination_output_dir = os.path.join(output_dir, 'illumination/met')
    make_dir(illumination_output_dir)
    
    model.eval()
    for i, image in enumerate(tqdm(test_loader, desc='\r')):#*  B,C,H,W
        name = test_loader.dataset.images_name[i]
        image = Variable(image, requires_grad=False)
        image = image.cuda()
        B,C,H,W = image.shape 
        trans1 = transforms.Compose([transforms.Resize(size=(H//16*16, W//16*16))])
        trans2 = transforms.Compose([transforms.Resize(size=(H, W))])
        image = trans1(image)#* debug

        with torch.no_grad():
            res= model(image)#* out_background,out_depth, out_normal, out_reflectance, out_illumination

        # out_edge = trans2(res[0])
        out_depth, out_normal, out_reflectance, out_illumination = trans2(res[1]),trans2(res[2]),trans2(res[3]),trans2(res[4])

        depth_pred = out_depth.data.cpu().numpy()
        depth_pred = depth_pred.squeeze()
        sio.savemat(os.path.join(depth_output_dir, '{}.mat'.format(name)), {'result': depth_pred})


        normal_pred = out_normal.data.cpu().numpy()
        normal_pred = normal_pred.squeeze()
        sio.savemat(os.path.join(normal_output_dir, '{}.mat'.format(name)), {'result': normal_pred})

        reflectance_pred = out_reflectance.data.cpu().numpy()
        reflectance_pred = reflectance_pred.squeeze()
        sio.savemat(os.path.join(reflectance_output_dir, '{}.mat'.format(name)), {'result': reflectance_pred})

        illumination_pred = out_illumination.data.cpu().numpy()
        illumination_pred = illumination_pred.squeeze()
        sio.savemat(os.path.join(illumination_output_dir, '{}.mat'.format(name)),
                    {'result': illumination_pred})
    
    #* Due to environment conflicts, activate another virtual environment in a separate shell to perform the evaluation.

    return eval_dir(output_dir)
    

def test_visul_label(label ,output_name):
    
    # print(label.shape)
    # print(type(label))
    # label =label[:, :, 0]

    cmap = np.array([[0, 0, 0],
    [128, 0, 0],
    [128, 128, 0],
    [0, 128, 0],
    [0, 0, 128]]
    )
    y = label
    r = y.copy()
    g = y.copy()
    b = y.copy()
    # print('r=',r)
    for l in range(0, len(cmap)):
        r[y == l] = cmap[l, 0]
        g[y == l] = cmap[l, 1]
        b[y == l] = cmap[l, 2]
    label=np.concatenate((np.expand_dims(b,axis=-1),np.expand_dims(g,axis=-1),
                            np.expand_dims(r,axis=-1)),axis=-1)

    cv2.imwrite(output_name,label)



def eval_dir(output_dir):
    tic = time.time()
    os.system("./eval_tools/test.sh %s %s"%(output_dir,"1"))
    spend_time =  time.time() - tic
    logger.info("validation spend time : "+time.strftime("%H:%M:%S",time.gmtime(spend_time)))
    
    
    with open (osp.join(output_dir,"eval_res.json"),'r')as f :
        eval_res = json.load(f)

    return eval_res
    

def eval_precompute_res():
    p = "/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/networks/precomputed/rindnet-resnet50"
    
    logger.info(eval_dir(p))
    
    

def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    #* load data 
    #!+=========================================
    # test_dataset = Mydataset(root_path=args.test_dir, split='test', crop_size=args.crop_size)
    # test_dataset = NYUD_GeoNet(split='val',root='/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/data/nyud2',mode='inference')
    # test_dataset=Nyud2()
    test_dataset = IIWDataset(data_dir='/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/data/IIW/iiw-dataset',split='test')

    #!+=========================================

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, 
                        shuffle=False,num_workers=args.workers,pin_memory=False)
    logger.info(args.run_id)
    logger.info(args.save_file)

    test_edge(args.resume,test_loader,args.save_file,args.run_id)#! The provided model path for resuming training should be an absolute path.
    
if __name__ == '__main__':
    main()
    # eval_precompute_res()

    