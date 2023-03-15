'''
Author: daniel
Date: 2023-02-07 12:50:58
LastEditTime: 2023-03-14 22:27:45
LastEditors: daniel
Description: 
FilePath: /Cerberus-main/dataloaders/datasets/sbu.py
have a nice day
'''





import os
import sys
import tarfile
import cv2

from PIL import Image
import numpy as np
import torch.utils.data as data
import scipy.io as sio
import h5py
from natsort import natsorted

from loguru import logger 
from torchvision import transforms
from tqdm import tqdm

import torch

from torchvision.transforms import ToTensor

# from utils.mypath import MyPath
# from util import mkdir_if_missing

import os.path as osp 



import scipy.io as scio
 

def make_dir(path):
    if not osp.exists(path):
        os.makedirs(path)


class SBU(data.Dataset):

    def __init__(self,path,subset='train'):
        self.subset=subset
        
        self.root=path

        self.subset_path = osp.join(path,'SBUTrain4KRecoveredSmall' if subset=='train' else 'SBU-Test')

        self.image_path = osp.join(self.subset_path ,'ShadowImages')
        self.mask_path = osp.join(self.subset_path ,'ShadowMasks')

        self.edge_path = osp.join(self.subset_path ,'EdgeMap')
        make_dir(self.edge_path)

        self.edge_mat_path = osp.join(self.subset_path ,'EdgeMapMat')
        make_dir(self.edge_mat_path)
        

        self.image_list = sorted(os.listdir(self.image_path))
        self.mask_list = sorted(os.listdir(self.mask_path ))
        logger.info(f" length : {len(self.image_list)},{len(self.mask_list)} ")
        
        logger.info(f'ready to process data')
        self.preprocess()
        logger.info(f'process done ')
        self.edge_list= sorted(os.listdir(self.edge_path ))
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.trans = transforms.Compose([transforms.ToTensor(),normalize])  ## pre-process of pre-trained model of pytorch resnet-50
        self.save_mat_for_eval()
        self.images_name = ['.'.join(x.split('.')[:-1]) for x in self.image_list]
        
        

    '''
    description:  detect edge using canny from opencv 
    param {*} self
    param {*} map
    param {*} low_threshold
    param {*} high_threshold
    return {*}
    '''
    def detect_edge(self,map,low_threshold=100,high_threshold=200):

        return cv2.Canny(np.uint8(map),low_threshold,high_threshold)

    

    '''
    description:  generate edge map from  shadow mask to edge map 
    param {*} self
    return {*}
    '''
    def preprocess(self):
        for mask_name in tqdm(self.mask_list ):
            save_path = osp.join(self.edge_path,mask_name)
            if not osp.exists(save_path):
                mask_img = self.imread(osp.join(self.mask_path,mask_name),gray=True)
                edge =self.detect_edge(mask_img)
                cv2.imwrite(save_path,edge)

    '''
    description:  save the edge map as mat file for evaluation 
    param {*} self
    return {*}
    '''
    def save_mat_for_eval(self):

        for edge_name in tqdm(self.edge_list ):
            
            #!=====================
            save_path = osp.join(self.edge_mat_path,'.'.join(edge_name.split('.')[:-1])+".mat")
            #!=====================

            if not osp.exists(save_path):
                edge = self.imread(osp.join(self.edge_path,edge_name),gray=True)/255
                self.save_as_mat(save_path,edge)
                
                
    
    def __len__(self):
        return len(self.image_list )
    

    '''
    description: read image by opencv
    param {*} self
    param {*} path
    param {*} gray
    return {*}
    '''
    def imread(self,path,gray =False):
        return cv2.imread(path,cv2.IMREAD_COLOR if not gray else cv2.IMREAD_GRAYSCALE)


    '''
    description:  read image by pillow
    param {*} self
    param {*} path
    param {*} gray
    return {*}
    '''
    def imread2(self,path,gray =False):
        return Image.open(path).convert('RGB' if not gray else 'L') 

    '''
    description:  save  mat file for evaluation 
    param {*} self
    param {*} file_name
    param {*} gt_map
    return {*}
    '''
    def save_as_mat(self,file_name,gt_map):

        scio.savemat(file_name,
            {'__header__': b'MATLAB 5.0 MAT-file, Platform: MACI64, Created on: Mon Feb 7 06:47:01 2023',
            '__version__': '1.0',
            '__globals__': [],
            'groundTruth': [{'Boundaries':gt_map}]
            }
        )
            
        

    '''
    description:  get item 
    param {*} self
    param {*} index
    return {*}
    '''
    def __getitem__(self, index):
        # logger.info(f" {self.image_list[index]} loaded ")
        if self.subset=='train':

            # img=self.imread(os.path.join(self.image_path,self.image_list[index]))
            # shadow_mask=self.imread(os.path.join(self.mask_path,self.mask_list[index]),gray=True)
            # edge_map = self.imread(os.path.join(self.edge_path,self.edge_list[index]),gray=True)

            img=self.imread2(os.path.join(self.image_path,self.image_list[index]))
            shadow_mask=self.imread2(os.path.join(self.mask_path,self.mask_list[index]),gray=True)
            edge_map = self.imread2(os.path.join(self.edge_path,self.edge_list[index]),gray=True)
            return self.trans(img), ToTensor()(edge_map).float() 
        else:
            return self.trans(self.imread2(os.path.join(self.image_path,self.image_list[index]))),self.images_name[index]


    def name2idx(self,name):
        return self.image_list.index(name)
    

    def getitem(self, index):
        img=self.imread(os.path.join(self.image_path,self.image_list[index]))
        shadow_mask=self.imread(os.path.join(self.mask_path,self.mask_list[index]),gray=True)

        edge_map = self.imread(os.path.join(self.edge_path,self.edge_list[index]),gray=True)

        return img, edge_map
    
    def getitem_all(self, index):
        img=self.imread(os.path.join(self.image_path,self.image_list[index]))
        shadow_mask=self.imread(os.path.join(self.mask_path,self.mask_list[index]),gray=True)

        edge_map = self.imread(os.path.join(self.edge_path,self.edge_list[index]),gray=True)

        return img, shadow_mask,edge_map
        
        


if __name__ == "__main__":
    dataset  = SBU(path='/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/data/SBU/SBU-shadow',subset='val')
    # dataset  = SBU(path='/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/data/SBU/SBU-shadow')

    sample= dataset.__getitem__(0)



    
    

        
        

        
        
    

        
        

        

        

        

        

        

        

        

        




