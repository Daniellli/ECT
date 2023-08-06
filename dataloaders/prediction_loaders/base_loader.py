import os 
from os.path import join,split,exists
import numpy as np 

import scipy.io as scio

import cv2


from utils.utils import * 
    



class BaseLoader:


    def __init__(self,path):

        
        self.root = path 

        self.mats = sorted([x for x in os.listdir(join(path,'met')) if x.endswith('mat')])


        self.names = [x.split('.')[0] for x in self.mats ]
        

        self.nmss = sorted([x for x in os.listdir(join(path,'nms')) if x.endswith('png')])

        self.nms_evals = sorted([x for x in os.listdir(join(path,'nms-eval')) if x.endswith('txt')])



        self.eval_bdry_img = np.loadtxt(join(path,'nms-eval','eval_bdry_img.txt'),delimiter = '\t', dtype=np.str0).tolist()


        self.eval_bdry_img = np.array([ y for x in self.eval_bdry_img for y in x.split(" ")  if len(y) != 0 ]).reshape([-1,5])

        self.eval_bdry_thr = np.loadtxt(join(path,'nms-eval', 'eval_bdry_thr.txt' ),delimiter = '\t', dtype=np.str0).tolist()

        self.eval_bdry_thr = np.array([ y for x in self.eval_bdry_thr for y in x.split(" ")  if len(y) != 0 ]).reshape([-1,4])

        self.eval_bdry = np.loadtxt(join(path,'nms-eval', 'eval_bdry.txt' ),delimiter = '\t', dtype=np.str0)
        

    

    def __len__(self):
        return len(self.names)
        

    def name2idx(self,name):
        return self.names.index(name)

    def getitem(self,idx):
        # print(self.names[idx])

        mat = np.array(load_mat(join(self.root,'met',self.mats[idx]))['result'])

        
        nms = cv2.imread(join(self.root,'nms',self.nmss[idx]),cv2.IMREAD_GRAYSCALE)

        # print(nms.shape)

        return mat,nms

        # eval_res = np.loadtxt(join(self.root,'nms-eval',self.nms_evals[idx]),delimiter = '\t', dtype=np.str0).tolist()
        # eval_res = np.array([ y for x in eval_res for y in x.split("   ")  if len(y) != 0 ]).reshape([-1,5])
        # print(eval_res)

        


        
        
        



        


    


    

    
    
        

        

if __name__ == '__main__':
    loader = BaseLoader('/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/networks/need2release/wo_cause_interaction_0/illumination')
    loader.getitem(0)

        
        

        
        

        