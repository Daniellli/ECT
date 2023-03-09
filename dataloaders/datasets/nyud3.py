'''
Author: daniel
Date: 2023-02-10 19:53:33
LastEditTime: 2023-03-06 08:06:29
LastEditors: daniel
Description: NYUD2 dataloader
FilePath: /Cerberus-main/dataloaders/datasets/nyud3.py
have a nice day
'''




from utils import * 

import torch
from os.path import join,split,exists

import numpy as np 

from tqdm import tqdm

import scipy.io as scio
# from dataloaders.datasets.nyud_geonet import *
from torchvision import transforms

import os 

import torch.nn.functional as F
 
def make_dir(path):
    if not exists(path):
        os.makedirs(path)

''' 
description:  对goal 的非零元素进行扩张, 扩张倍数为10 
param {*} goal 
param {*} times
return {*}
'''
def dilation(goal, times = 2 ):
    selem = skimage.morphology.disk(times)


    # goal = skimage.morphology.binary_dilation(goal, selem) != True
    goal = morphology.binary_dilation(goal, selem) != True
    goal = 1 - goal * 1.
    goal*=255
    return goal



def crop_save(path,src):
        if not exists(path):
            imwrite(path,src[45:471, 41:601])


'''
description:  save  mat file for evaluation 
param {*} self
param {*} file_name
param {*} gt_map
return {*}
'''
def save_as_mat(file_name,gt_map):

    scio.savemat(file_name,
        {'__header__': b'MATLAB 5.0 MAT-file, Platform: MACI64, Created on: Mon Feb 7 06:47:01 2023',
        '__version__': '1.0',
        '__globals__': [],
        'groundTruth': [{'Boundaries':gt_map}]
        }
    )




def readtxt(path):
    return np.loadtxt(path,dtype=np.str0,delimiter='\n')


class Nyud3:

    def __init__(self,
                path='/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/data/nyud2',
                subset='NYU_origin',gen_edge=False):

        self.subset = subset
        self.path = join(path,subset)
        self.name_list = readtxt(join(self.path,'val.txt'))
        
        
        self.images_path = join(self.path,'nyu_images')
        self.depths_path = join(self.path,'nyu_depths')
        self.labels_path = join(self.path,'nyu_labels')
        
        self.normals_path = join(self.path,'nyu_normals') 
        self.edge_3x3_path = join(self.path,'nyu_edges_local_3x3') #* 3 X 3 actually
        self.edge_path = join(self.path,'nyu_edges_canny')

        make_dir(self.edge_path)
        # make_dir(self.normal_edge_path)

        self.edge_for_depth_normal='depth_normal_edges_3x3'
        # self.edge_for_depth_normal='depth_normal_edges_canny'
        # self.strategy = 'threashold_decay'
        # self.strategy = 'local_neighbors'
        self.strategy = 'range123'
        
        
        # strategy = 'local_neighbors'
        self.depth_edge_path = join(self.path,self.edge_for_depth_normal,self.strategy,'nyu_depth_edges')
        self.depth_edge_mat_path = join(self.path,self.edge_for_depth_normal,self.strategy,'nyu_depth_edges_mat')
        self.depth_edge_mat_crop_path = join(self.path,self.edge_for_depth_normal,self.strategy,'nyu_depth_edges_crop_mat')

        self.normal_edge_path = join(self.path,self.edge_for_depth_normal,self.strategy,'nyu_normal_edges')
        self.normal_edge_mat_path = join(self.path,self.edge_for_depth_normal,self.strategy,'nyu_normal_edges_mat')
        self.normal_edge_mat_crop_path = join(self.path,self.edge_for_depth_normal,self.strategy,'nyu_normal_edges_crop_mat')

        make_dir(self.depth_edge_path)
        make_dir(self.depth_edge_mat_path)
        make_dir(self.depth_edge_mat_crop_path)
        make_dir(self.normal_edge_path)
        make_dir(self.normal_edge_mat_path)
        make_dir(self.normal_edge_mat_crop_path)

        
        #* generate the normal map, crop the normal edge map of geonet and move them here actually.
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transformer = transforms.Compose([
                             transforms.ToTensor(),
                             normalize])  ## pre-process of pre-trained model of pytorch resnet-50

        
        # process_mp(self.gen_edge,range(self.__len__()),num_threads=256)
        # print(f"edge generation done ")

        if gen_edge:
            
            # self.gen_depth_normal_edge(0)
            # process_mp(self.gen_depth_normal_edge_threshold_decay,range(self.__len__()),num_threads=512)
            #* tmp actually.
            # process_mp(self.gen_depth_normal_edge_local_neighbors,range(self.__len__()),num_threads=256)
            # self.gen_depth_normal_edge_local_neighbors(0)
            pass

        
        # process_mp(self.fix_bug,range(self.__len__()),num_threads=512)
        
            
    def get_depth_normal_edge_genation_stategy(self):
        return self.edge_for_depth_normal+' + '+self.strategy

    '''
    description:  there is a bug existing in mat file, trun [0,255] to [0,1]
    param {*} self
    param {*} idx
    return {*}
    '''
    def fix_bug(self,idx):
        def fix(path):
            data = load_mat(path)
            data=data['groundTruth'][0,0]['Boundaries'][0,0]/255
            save_as_mat(path,data)

        
        depth_mat_path =  join(self.depth_edge_mat_path,self.name_list[idx].split('.')[0]+'.mat')
        depth_mat_crop_path =  join(self.depth_edge_mat_crop_path,self.name_list[idx].split('.')[0]+'.mat')

        normal_mat_path =  join(self.normal_edge_mat_path,self.name_list[idx].split('.')[0]+'.mat')
        normal_mat_crop_path =  join(self.normal_edge_mat_crop_path,self.name_list[idx].split('.')[0]+'.mat')
        
        fix(depth_mat_path)
        fix(depth_mat_crop_path)
        fix(normal_mat_path)
        fix(normal_mat_crop_path)
     
        
        
        
    
        
    def gen_edge(self,idx):

        save_path = join(self.edge_path,self.name_list[idx])
        if not exists(save_path):            
            label = imread(join(self.labels_path,self.name_list[idx]),gray=True)
            label=to_3_channel(normalize(label)*255).astype(np.uint8)
            edge_map = detect_edge(label,50,200)

            imwrite(save_path,edge_map)
            # show_imgs([detect_edge(label,50,150)*255],[0,0,])
            # print('hello world')



    def get_local_area(self,x,y,x_range,y_range,data):
        x_bound,y_bound=data.shape[:2]

        #* get local map 
        x_min = x-x_range
        x_max = x+x_range+1
        y_min = y-y_range
        y_max = y+ y_range+1

        
        if x_min <0:
            x_min=0
        elif x_max >= x_bound:
            x_max=x_bound-1
        
        if y_min < 0:
            y_min = 0
        elif y_max >= y_bound:
            y_max = y_bound-1

        return data[x_min:x_max,y_min:y_max]
        

    def gen_depth_normal_edge_threshold_decay(self,idx,depth_x_range = 2,depth_y_range = 2,depth_thrs = 4,
                        normal_x_range = 4,normal_y_range = 4,normal_thrs = 0.08):
        image,label,edge,depth_map,normal_map,name=self.getitem(idx)
        
        depth_edge = np.zeros_like(edge)
        normal_edge = np.zeros_like(edge)
        edge_idx  = np.argwhere(edge==255)

        def gen_edge(indexes):
            rest_edge_pixel= []
            for x,y in indexes:
                #* justify the depth edge 
                local_map = self.get_local_area(x,y,depth_x_range,depth_y_range,depth_map)
                dist = local_map.max() - local_map.min()
                if dist > depth_thrs:
                    depth_edge[x,y] = 255
                    continue

                #* justify the normal edge 
                local_map = torch.from_numpy(self.get_local_area(x,y,normal_x_range,normal_y_range,normal_map)).reshape([-1,3]).float()
                max_dist= 0
                for idx,z in enumerate(local_map):
                    dist = (1- F.cosine_similarity(local_map, z, dim=-1)).max()
                    # print('%d-th '%(idx), dist)
                    if max_dist < dist:
                        max_dist = dist
                # print(f'max distance : {max_dist}')
                if max_dist > normal_thrs:
                    normal_edge[x,y] = 255
                else:
                    rest_edge_pixel.append([x,y])
            return rest_edge_pixel

        rest_edge_pixel=gen_edge(edge_idx)
        # print('threshold decay, depth_thrs:%.4f, normal_thrs: %.4f,the pixel remaining: %d'%(depth_thrs,normal_thrs,len(rest_edge_pixel)))
        #* process the rest pixels 
        cnt = 0
        while len(rest_edge_pixel)!= 0 :
            if cnt == 80:
                # print(f'break')
                break 
            depth_thrs-=(depth_thrs*0.15)
            normal_thrs-=(normal_thrs*0.1)
            # print('%d-th,threshold decay, depth_thrs:%.4f, normal_thrs: %.4f,the pixel remaining: %d'%(cnt,depth_thrs,normal_thrs,len(rest_edge_pixel)))
            rest_edge_pixel=gen_edge(rest_edge_pixel)
            cnt+=1
            
        
        #* save  edges 
        #todo crop: [45:471, 41:601]
        depth_path =  join(self.depth_edge_path,self.name_list[idx])
        depth_mat_path =  join(self.depth_edge_mat_path,self.name_list[idx].split('.')[0]+'.mat')
        depth_mat_crop_path =  join(self.depth_edge_mat_crop_path,self.name_list[idx].split('.')[0]+'.mat')
        imwrite(depth_path,depth_edge)
        save_as_mat(depth_mat_path,depth_edge)
        save_as_mat(depth_mat_crop_path,depth_edge[45:471, 41:601])

        normal_path =  join(self.normal_edge_path,self.name_list[idx])
        normal_mat_path =  join(self.normal_edge_mat_path,self.name_list[idx].split('.')[0]+'.mat')
        normal_mat_crop_path =  join(self.normal_edge_mat_crop_path,self.name_list[idx].split('.')[0]+'.mat')
        imwrite(normal_path,normal_edge)
        save_as_mat(normal_mat_path,normal_edge)        
        save_as_mat(normal_mat_crop_path,normal_edge[45:471, 41:601])

        return depth_edge,normal_edge,rest_edge_pixel
        
        
        

        
    
    def gen_depth_normal_edge_local_neighbors(self,idx,depth_x_range = 2,depth_y_range = 2,depth_thrs = 4,
                        normal_x_range = 4,normal_y_range = 4,normal_thrs = 0.08):
       

        def gen_edge(indexes):
            rest_edge_pixel= []
            for x,y in indexes:
                #* justify the depth edge 
                local_map = self.get_local_area(x,y,depth_x_range,depth_y_range,depth_map)
                dist = local_map.max() - local_map.min()
                if dist > depth_thrs:
                    depth_edge[x,y] = 255
                    continue

                #* justify the normal edge 
                local_map = torch.from_numpy(self.get_local_area(x,y,normal_x_range,normal_y_range,normal_map)).reshape([-1,3]).float()
                max_dist= 0
                for idx,z in enumerate(local_map):
                    dist = (1- F.cosine_similarity(local_map, z, dim=-1)).max()
                    # print('%d-th '%(idx), dist)
                    if max_dist < dist:
                        max_dist = dist
                # print(f'max distance : {max_dist}')
                if max_dist > normal_thrs:
                    normal_edge[x,y] = 255
                else:
                    rest_edge_pixel.append([x,y])
            return rest_edge_pixel


        '''
        description:  
        param {*} indexes
        param {*} local_range : decide the pixel belong to depth or normal edge 
        return {*}
        '''
        def gen_edge_by_local_area(indexes,local_range=2):
            rest_edge_pixel2= []
            depth_cnt = normal_cnt = 0
            for x,y in indexes:
                
                local_map = self.get_local_area(x,y,local_range,local_range,depth_edge)
                if local_map.max() !=0:
                    depth_edge[x,y]=255
                    depth_cnt+=1
                    continue

                local_map = self.get_local_area(x,y,local_range,local_range,normal_edge)
                if local_map.max() !=0:
                    normal_edge[x,y]=255
                    normal_cnt+=1
                else:
                    rest_edge_pixel2.append([x,y])

            # print('local_range:%d; assign %d to depth and %d to normal; the pixel remaining: %d'%(local_range,depth_cnt,normal_cnt,len(rest_edge_pixel2)))
            return rest_edge_pixel2

        #* the file name of the final saved file
        #*============================================================================
        depth_path =  join(self.depth_edge_path,self.name_list[idx])
        depth_mat_path =  join(self.depth_edge_mat_path,self.name_list[idx].split('.')[0]+'.mat')
        depth_mat_crop_path =  join(self.depth_edge_mat_crop_path,self.name_list[idx].split('.')[0]+'.mat')

        normal_path =  join(self.normal_edge_path,self.name_list[idx])
        normal_mat_path =  join(self.normal_edge_mat_path,self.name_list[idx].split('.')[0]+'.mat')
        normal_mat_crop_path =  join(self.normal_edge_mat_crop_path,self.name_list[idx].split('.')[0]+'.mat')
        
        if exists(depth_path) and  exists(depth_mat_path) and  exists(depth_mat_crop_path) and  exists(normal_path) \
            and  exists(normal_mat_path) and exists(normal_mat_crop_path) :
            
            return 
        #*============================================================================
        

        

        image,label,edge,depth_map,normal_map,name=self.getitem(idx)
        
        depth_edge = np.zeros_like(edge)
        normal_edge = np.zeros_like(edge)

        edge_idx  = np.argwhere(edge==255)

        rest_edge_pixel=gen_edge(edge_idx)
        # print('threshold decay, depth_thrs:%.4f, normal_thrs: %.4f,the pixel remaining: %d'%(depth_thrs,normal_thrs,len(rest_edge_pixel)))

        #* process the rest pixels 
        # local_range=1
        # while len(rest_edge_pixel)!=0:
        #     rest_edge_pixel = gen_edge_by_local_area(rest_edge_pixel,local_range=local_range)
        #     local_range+=1
        # print('final local range: %d'%(local_range-1))
        #* modify as following shown :
        rest_edge_pixel = gen_edge_by_local_area(rest_edge_pixel,local_range=1)
        rest_edge_pixel = gen_edge_by_local_area(rest_edge_pixel,local_range=2)
        rest_edge_pixel = gen_edge_by_local_area(rest_edge_pixel,local_range=3)

        #* 全部归类完需要10 range.... from 1 to 10 .... 
        #* save 
        #*============================================================================
        imwrite(depth_path,depth_edge)
        depth_edge = depth_edge/255
        save_as_mat(depth_mat_path,depth_edge)
        save_as_mat(depth_mat_crop_path,depth_edge[45:471, 41:601])

        imwrite(normal_path,normal_edge)
        normal_edge = normal_edge/255
        save_as_mat(normal_mat_path,normal_edge)        
        save_as_mat(normal_mat_crop_path,normal_edge[45:471, 41:601])
        #*============================================================================
            
        return depth_edge,normal_edge,rest_edge_pixel
        

        

    '''
    description:  gen one edge map 
    param {*} self
    param {*} idx
    return {*}
    '''
    def gen_save_depth_edge(self,idx,depth_threshold = 2):
        
        
        # save_path =  join(self.depth_edge_path,self.name_list[idx])
        save_path =  join(self.__depth_edge_path2,self.name_list[idx])
        mat_save_path =  join(self.__depth_edge2_mat_path,self.name_list[idx].split('.')[0]+'.mat')
        

        if not  exists(save_path) :    

            _,depth_map,_,edge_map,_,_,_,_=self.getitem_no_crop(idx)
            # image,depth,label,edge,depth_edge,normal_map,normal_edge,self.name_list[idx]

            #* dilation edge,  
            # edge_map= dilation(edge_map,2)/255

            depth_edge =get_depth_edge_by_edge(depth_map,edge_map,depth_threshold=depth_threshold)*255
            
            # imwrite(save_path,depth_edge)
            #* save the cropped depth edge 
            crpped = depth_edge[45:471, 41:601]
            imwrite(save_path,crpped)
            self.save_as_mat(mat_save_path,crpped)


            # imwrite('a.jpg',depth_map)
            # imwrite('b.jpg',edge_map)
            # imwrite('c.jpg',crpped)




    def __len__(self):
        return len(self.name_list)

    
    def name2idx(self,name):
        return self.name_list.index(name)



    def getitem(self,idx):
        
        
        image = imread(join(self.images_path,self.name_list[idx]))
        label = imread(join(self.labels_path,self.name_list[idx]),gray=True)
        # edge = imread(join(self.edge_path,self.name_list[idx]),gray=True)
        edge = imread(join(self.edge_3x3_path,self.name_list[idx]),gray=True)

        depth_map = imread(join(self.depths_path,self.name_list[idx]),gray=True)
        normal_map = imread(join(self.normals_path,self.name_list[idx]))


        return image,label,edge,depth_map,normal_map,self.name_list[idx]
    

    def getitem_all(self,idx):
        
        
        image = imread(join(self.images_path,self.name_list[idx]))
        label = imread(join(self.labels_path,self.name_list[idx]),gray=True)
        # edge = imread(join(self.edge_path,self.name_list[idx]),gray=True)

        if self.edge_for_depth_normal=='depth_normal_edges_canny':
            edge = imread(join(self.edge_path,self.name_list[idx]),gray=True)
        else :
            edge = imread(join(self.edge_3x3_path,self.name_list[idx]),gray=True)

        

        depth_map = imread(join(self.depths_path,self.name_list[idx]),gray=True)
        normal_map = imread(join(self.normals_path,self.name_list[idx]))

        depth_edge = imread(join(self.depth_edge_path,self.name_list[idx]),gray=True)
        normal_edge = imread(join(self.normal_edge_path,self.name_list[idx]),gray=True)

        
        return image,label,edge,depth_map,depth_edge,normal_map,normal_edge,self.name_list[idx]

    def getitem_by_name(self,name):
        image = imread(join(self.images_path,name))
        label = imread(join(self.labels_path,name),gray=True)
        # edge = imread(join(self.edge_path,name),gray=True)

        if self.edge_for_depth_normal=='depth_normal_edges_canny':
            edge = imread(join(self.edge_path,name),gray=True)
        else :
            edge = imread(join(self.edge_3x3_path,name),gray=True)

        

        depth_map = imread(join(self.depths_path,name),gray=True)
        normal_map = imread(join(self.normals_path,name))

        depth_edge = imread(join(self.depth_edge_path,name),gray=True)
        normal_edge = imread(join(self.normal_edge_path,name),gray=True)


        return image,label,edge,depth_map,depth_edge,normal_map,normal_edge,name



    def __getitem__(self,idx):

        image = imread(join(self.images_path,self.name_list[idx]))


        
        # return self.transformer(image),self.name_list[idx].split('.')[0]
        return self.transformer(image[45:471, 41:601]),self.name_list[idx].split('.')[0]


if __name__ == "__main__":
    # Nyud3(gen_edge=True)
    Nyud3()
    