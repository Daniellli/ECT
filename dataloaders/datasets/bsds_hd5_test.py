import os
import torch
import h5py
import random
import numpy as np
import scipy.io
from tqdm import tqdm
from torch.utils.data import DataLoader
from PIL import Image,ImageStat
from torch.utils import data
from torchvision import transforms
import torch
import numpy as np
from IPython import embed

from PIL import Image
import os 
from os.path import join, split, isdir,isfile
from torchvision import transforms
import matplotlib.pyplot as plt
from utils import *



def load_mat_gt(data_path):
    gt  = load_mat(data_path)
    
    return gt['groundTruth'][0,0]['Boundaries'][0,0]
    




class MydatasetTest(torch.utils.data.Dataset):

    def __init__(self,root_path='....../Augmentation/'):
        list_file = os.path.join(root_path, 'test.lst')
        self.data_root = root_path

        # self.images_path = np.loadtxt(list_file,dtype=np.str0)
        
        with open(list_file, 'r') as f:
            lines = f.readlines()
        lines = [line.strip() for line in lines]
        self.images_path = lines
        
        self.images_name = []
        for path in self.images_path:
            folder, filename = os.path.split(path)
            name, ext = os.path.splitext(filename)
            self.images_name.append(name)
        
        
        #* preprocesser for image
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.trans = transforms.Compose([
                             transforms.ToTensor(),
                             normalize])  
        
        """
        including DNRI, depth, normal, reflectance, illumination 
        """
    def getitem_label(self,idx):
        
        # task_list = ['all_edges','depth','normal','reflectance','illumination']
        task_list = ['depth','normal','reflectance','illumination']
        # print(self.images_name[idx])
        gt_list = []
        
        for task in task_list:
            # print(task)
            gt_list.append(load_mat_gt(join(self.data_root,'../../testgt/%s'%(task),self.images_name[idx]+'.mat')))
            
            
        return gt_list
        

    def idx2name(self,idx):
        
        return self.images_name[idx]

    def name2idx(self,name):
        return self.images_name.index(name)

    def __len__(self):
        return len(self.images_path)


    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.images_path[idx])).convert('RGB')
        img_tensor = self.trans(img)
        

        gt_list = self.getitem_label(idx)

        label = np.concatenate([x[None,...] for x in gt_list],axis=0)
        
        #* dnri  as the train schedule do 
        label = torch.from_numpy(label).float()
    
        return img_tensor,label
        # return img_tensor
    
    



if __name__ == '__main__':
    test_dataset = MydatasetTest(root_path='data/BSDS-RIND/BSDS-RIND/Augmentation/')
    
    tbar = tqdm(test_dataset)

    for i, (image, target) in enumerate(tbar):
        print(image.shape,target.shape)







