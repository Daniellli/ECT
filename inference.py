'''
Author: xushaocong
Date: 2022-09-05 20:50:42
LastEditTime: 2022-09-11 19:03:17
LastEditors: xushaocong
Description: 
FilePath: /Cerberus-main/inference.py
email: xushaocong@stu.xmu.edu.cn
'''
'''
Author: xushaocong
Date: 2022-06-20 22:49:32
LastEditTime: 2022-08-30 09:31:42
LastEditors: xushaocong
Description: 
FilePath: /Cerberus-main/test.py
email: xushaocong@stu.xmu.edu.cn
'''


import cv2

import os
# from IPython import embed #for terminal debug 
# import pdb
import time
from cv2 import sort
import numpy as np
import sys
import torch
import torch.backends.cudnn as cudnn
from torchvision import  transforms
from torch.autograd import Variable
import data_transforms as transforms
from model.edge_model import  EdgeCerberus
import torch 
import os.path as osp
from tqdm import tqdm
import scipy.io as sio
import torchvision.transforms as transforms

from IPython  import embed
import glob
from loguru import logger
from dataloaders.datasets.bsds_hd5 import Mydataset
from utils import make_dir
import argparse
import json
import warnings
warnings.filterwarnings('ignore')

import json
from utils.global_var import *

from torch.utils.data import Dataset


from plot.draw_inference import draw,draw_grid2

import math

def my_transform(image,to_size):
    return torch.nn.functional.interpolate(image,size=to_size)

'''
description: 
param {*} args
return {*}
'''
def inference(model_abs_path,test_loader,src_data_dir,runid=None,):
    tic = time.time()
    if runid is  None:
        # output_dir  =osp.join(osp.dirname(src_data_dir),time.strftime("%Y:%m:%d",time.gmtime(time.time()))+"_"+str(int(time.time())))
        output_dir  =osp.join(osp.dirname(src_data_dir),'nms_res')
    else:
        # output_dir  =osp.join(osp.dirname(src_data_dir),time.strftime("%Y:%m:%d",time.gmtime(time.time()))+"_"+str(int(time.time()))+"_"+str(runid))
        output_dir  =osp.join(osp.dirname(src_data_dir),'nms_res'+"_"+str(runid))

        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    #* 加载模型
    single_model = EdgeCerberus(backbone="vitb_rn50_384",enable_attention_hooks=True)
    
    checkpoint = torch.load(model_abs_path,map_location='cuda:0')
    for name, param in checkpoint['state_dict'].items():
        name = name.replace("module.","") #* 因为分布式训练的原因导致多封装了一层
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



    #* 判断一些是否测试过了 , 测试过就不重复测试了
    
    if not(len(glob.glob(normal_output_dir+"/*.mat")) == len(test_loader)): 
        model.eval()
        tbar = tqdm(test_loader, desc='\r')
        
        for i, image in enumerate(tbar):#*  B,C,H,W
            
            name = test_loader.dataset.images_name[i]
            image = Variable(image, requires_grad=False)
            image = image.cuda()
            image = image.permute([0,3,1,2]).float()
            B,C,H,W = image.shape 
            to_size = (math.ceil(H/32)*32,math.ceil(W/32)*32)
            # logger.info(f'origin size = {(H,W)}, to size : {to_size}')
            trans1 = transforms.Compose([transforms.Resize(size= to_size )])#* 需要是32的倍数, 
            trans2 = transforms.Compose([transforms.Resize(size=(H, W))])
            image = trans1(image)#* debug


            with torch.no_grad():
                res= model(image)#* out_background,out_depth, out_normal, out_reflectance, out_illumination

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

            

    
    logger.info("reference done  ")
    #* nms
    os.system("bash eval_tools/inference.sh %s"%(output_dir))
    logger.info("nms done  ")
    spend_time =  time.time() - tic
    logger.info("spend time : "+time.strftime("%H:%M:%S",time.gmtime(spend_time)))
    #* plot 
    draw_grid2(src_data_dir,output_dir)

        


def args_parsing():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d', '--data-dir')
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--workers', default=40, type=int)
    parser.add_argument('--gpu-ids', default='7', type=str)
    parser.add_argument("--run-id", type=int,default=None,help="for evaluation ")
    
    
    return parser.parse_args()



class InferDatasets(Dataset):


    def __init__(self,data_path) -> None:
        super().__init__()
        self.data_path = data_path
        self.__get_list()
        self.__get_img_names()
        
        

    def __get_img_names(self):
        
        self.images_name =  sorted([x.split('.')[0] for x in os.listdir(self.data_path) if x.endswith(self.suffix)])
        
    def __get_list(self):
        self.suffix = "png"
        self.imgs_list = sorted(glob.glob(self.data_path+"/*."+self.suffix))
        
        if len(self.imgs_list) == 0 :
            self.suffix='jpg'
            self.imgs_list = sorted(glob.glob(self.data_path+"/*."+self.suffix))
            


        

    def __len__(self):
        return len(self.imgs_list)



    def __getitem__(self, index):
        img_path = self.imgs_list[index]
        return (cv2.imread(img_path))
        
        

def main():
    args = args_parsing()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    #* load data 
    test_dataset = InferDatasets(data_path=args.data_dir)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, 
                        shuffle=False,num_workers=args.workers,pin_memory=True)
    logger.info(args.run_id)
    logger.info(args.data_dir)
    inference(args.resume,test_loader,args.data_dir,args.run_id)#! resume 给的model path需要是绝对路径


    
if __name__ == '__main__':
    main()
    

    