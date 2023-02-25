'''
Author: daniel
Date: 2023-02-10 19:53:33
LastEditTime: 2023-02-20 16:22:38
LastEditors: daniel
Description: NYUD2 dataloader
FilePath: /Cerberus-main/dataloaders/datasets/nyud2.py
have a nice day
'''




from utils import * 

from os.path import join,split,exists

import numpy as np 

from tqdm import tqdm

import scipy.io as scio
# from dataloaders.datasets.nyud_geonet import *
from torchvision import transforms

import os 


 
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


class Nyud2:

    def __init__(self,
                path='/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/data/nyud2',
                subset='NYU_origin',
                gen_edge = False):

        self.subset = subset
        self.path = join(path,subset)

        self.name_list= readtxt(join(self.path,'val.txt'))
        
        
        self.depths_path = join(self.path,'nyu_depths')
        self.images_path = join(self.path,'nyu_images')
        self.labels_path = join(self.path,'nyu_labels')
        self.normals_path = join(self.path,'nyu_normals')

        self.edge_path = join(self.path,'nyu_edges')
        self.depth_edge_path = join(self.path,'nyu_depth_edge_canny')
        self.normal_edge_path = join(self.path,'nyu_normal_edge_canny')


        make_dir(self.edge_path)
        make_dir(self.depth_edge_path)
        make_dir(self.normals_path)
        make_dir(self.normal_edge_path)
        
        
        #* the data need to crop 
        #* 1. depth edge (canny); 2. image; 3.  normal edge (canny); 
        #* 4. depth map; 5. normal  map 
        #* target scale : [45:471, 41:601]
        
        self.cropped_path = join(self.path,'cropped')

        self.__depth_edge_path= join(self.cropped_path,'nyu_depth_edge_canny')
        self.__depth_edge_path2= join(self.cropped_path,'nyu_depth_edge')
        self.__image_path= join(self.cropped_path,'nyu_images')
        self.__normal_edge_path= join(self.cropped_path,'nyu_normal_edge_canny')
        self.__depth_path= join(self.cropped_path,'nyu_depths')
        self.__normal_path= join(self.cropped_path,'nyu_normals')

        self.__depth_edge_mat_path= join(self.cropped_path,'nyu_depth_edge_canny_mat')
        self.__depth_edge2_mat_path= join(self.cropped_path,'nyu_depth_edge_mat')
        self.__normal_edge_mat_path= join(self.cropped_path,'nyu_normal_edge_canny_mat')

        make_dir(self.__depth_edge_path)
        make_dir(self.__depth_edge_path2)
        make_dir(self.__depth_edge2_mat_path)
        
        make_dir(self.__image_path)
        make_dir(self.__normal_edge_path)
        make_dir(self.__depth_path)
        make_dir(self.__normal_path)
        make_dir(self.__depth_edge_mat_path)
        make_dir(self.__normal_edge_mat_path)
        

        if gen_edge:
            self.gen_edge_mp()
            self.gen_depth_edge_mp()
            # self.gen_normal_edge_mp()

            #* cropped data generation
            # self.gen_cropped_data()
        
        #* generate the normal map, crop the normal edge map of geonet and move them here actually.
        # self.geonet_loader = NYUD_GeoNet(split='val',root='/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/data/nyud2')
        # self.gen_normal_mp()
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transformer = transforms.Compose([
                             transforms.ToTensor(),
                             normalize])  ## pre-process of pre-trained model of pytorch resnet-50

        # self.gen_mat_mp()



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


    def gen_mat(self,idx):

        image,depth_map,label,edge,depth_edge,normal_map,normal_edge,name =self.getitem(idx)
        self.save_as_mat(join(self.__depth_edge_mat_path,name.split('.')[0]+'.mat'),depth_edge)
        self.save_as_mat(join(self.__normal_edge_mat_path,name.split('.')[0]+'.mat'),normal_edge)

        imwrite('a.jpg',depth_edge)
        imwrite('b.jpg',normal_edge)

        
    def gen_mat_mp(self):
        # process_mp(self.gen_mat,range(self.__len__()),num_threads=256)
        process_mp(self.gen_mat,range(self.__len__()),num_threads=1)
        



    def gen_one_cropped_data(self,idx):

        def crop_save(path,src):

            if not exists(path):
                imwrite(path,src[45:471, 41:601])

        image,depth_map,label,edge,depth_edge,normal_map,normal_edge,name =self.__getitem__(idx)
        #*1.  save image 
        crop_save(join(self.__image_path,name),image)

        #*2.  save depth edge 
        crop_save(join(self.__depth_edge_path,name),depth_edge)


        #*3.  save normal edge 
        crop_save(join(self.__normal_edge_path,name),normal_edge)


        #*4.  save  depth map 
        crop_save(join(self.__depth_path,name),depth_map)

        #*5.  save normal map
        crop_save(join(self.__normal_path,name),normal_map)
        
    
    '''
    description: 
    param {*} self
    return {*}
    '''
    def gen_cropped_data(self):
        process_mp(self.gen_one_cropped_data,range(self.__len__()),num_threads=256)
        


    '''
    description: generate one  normal map 
    param {*} self
    param {*} idx
    return {*}
    '''
    def gen_one_normal(self,idx):

        # for idx,(sample,name) in tqdm(enumerate(geonet_loader)):
        sample,_=self.geonet_loader.__getitem__(idx)
        _,_,_,_,_,_,_,name=self.__getitem__(idx)

        # image,_,_,_,_,_,_,name=self.__getitem__(idx)
        matched_normal = convert_image_vertical((normal_reverse_process(sample['normals'])*255).astype(np.uint8))
        imwrite(join(self.normals_path,name),matched_normal)
        # imwrite('a.jpg',image)
        # imwrite('b.jpg',matched_normal)

    '''
    description: generate the normal map 
    param {*} self
    return {*}
    '''
    def gen_normal_mp(self):
        process_mp(self.gen_one_normal,range(self.__len__()),num_threads=256)
        
            
        

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


    '''
    description:  gen one depth edge 
    param {*} self
    param {*} idx
    return {*}
    '''
    def gen_save_depth_edge_by_canny(self,idx):
        
        save_path =  join(self.depth_edge_path,self.name_list[idx])

        if not  exists(save_path):
            _,depth,_,edge_map,_,_,_,_=self.__getitem__(idx)

              
            depth_edge = detect_edge(depth,15,55)
            #* 1. gen by edge map and local 5 X 5 depth value
            edge_map= dilation(edge_map,10)
            depth_edge[edge_map!=255]=0

            imwrite(save_path,depth_edge)

    

    def gen_depth_edge_mp(self):
        process_mp(self.gen_save_depth_edge,range(self.__len__()),num_threads=256)
        # process_mp(self.gen_save_depth_edge_by_canny,range(self.__len__()),num_threads=1)
        

    '''
    description:  gen one normal edge map 
    param {*} self
    param {*} idx
    return {*}
    '''
    def gen_save_normal_edge_by_canny(self,idx):
        
        save_path =  join(self.normal_edge_path,self.name_list[idx])

        if not  exists(save_path) :
            _,_,_,edge_map,_,normal_map,_,_=self.__getitem__(idx)

            normal_edge = detect_edge(normal_map,50,150)

            #* 1. gen by edge map and local 5 X 5 depth value
            # edge_map= dilation(edge_map,2)            
            # normal_edge[edge_map!=255]=0
            imwrite(save_path,normal_edge)

            # imwrite('a.jpg',normal_edge)
            # imwrite('b.jpg',edge_map)

    '''
    description: generate the normal map 
    param {*} self
    return {*}
    '''
    def gen_normal_edge_mp(self):
        # process_mp(self.gen_one_normal,range(self.__len__()),num_threads=256)
        process_mp(self.gen_save_normal_edge_by_canny,range(self.__len__()),num_threads=256)
        
            

    '''
    description:  gen one edge map 
    param {*} self
    param {*} idx
    return {*}
    '''
    def gen_save_edge(self,idx):
        save_path =  join(self.edge_path,self.name_list[idx])

        if not  exists(save_path) :    
            label = imread(join(self.labels_path,self.name_list[idx]),gray=True)
            edge_map=get_edge_map_from_label(label)
            imwrite(save_path,edge_map*255)
        
    
    '''
    description:  generate edge multi thread
    param {*} self
    return {*}
    '''
    def gen_edge_mp(self):
        process_mp(self.gen_save_edge,range(self.__len__()),num_threads=256)
        

    

    def __len__(self):
        return len(self.name_list)

    

    def getitem_no_crop(self,idx):
        
        # return self.getitem_PIL(idx)
        depth = imread(join(self.depths_path,self.name_list[idx]),gray=True)
        image = imread(join(self.images_path,self.name_list[idx]))
        label = imread(join(self.labels_path,self.name_list[idx]),gray=True)
        edge = imread(join(self.edge_path,self.name_list[idx]),gray=True)


        depth_edge = imread(join(self.depth_edge_path,self.name_list[idx]),gray=True)
        normal_map = imread(join(self.normals_path,self.name_list[idx]))

        normal_edge  = imread(join(self.normal_edge_path,self.name_list[idx]),gray=True)

        return image,depth,label,edge,depth_edge,normal_map,normal_edge,self.name_list[idx]




    def __getitem__(self,idx):
        
        # return self.getitem_PIL(idx)
        image = imread(join(self.__image_path,self.name_list[idx]))
        depth = imread(join(self.__depth_path,self.name_list[idx]),gray=True)

        #* no cropped ========================================================
        label = imread(join(self.labels_path,self.name_list[idx]),gray=True)
        edge = imread(join(self.edge_path,self.name_list[idx]),gray=True) 
        #* ==================================================================


        depth_edge = imread(join(self.__depth_edge_path,self.name_list[idx]),gray=True)
        normal_map = imread(join(self.__normal_path,self.name_list[idx]))
        normal_edge  = imread(join(self.__normal_edge_path,self.name_list[idx]),gray=True)

        
        # return image,depth,label,edge,depth_edge,normal_map,normal_edge,self.name_list[idx]
        # return self.transformer(image),depth,label,edge,depth_edge,normal_map,normal_edge,self.name_list[idx].split('.')[0]
        return self.transformer(image),self.name_list[idx].split('.')[0]

    

    def getitem(self,idx):
        
        # return self.getitem_PIL(idx)
        image = imread(join(self.__image_path,self.name_list[idx]))
        depth = imread(join(self.__depth_path,self.name_list[idx]),gray=True)

        #* no cropped ========================================================
        label = imread(join(self.labels_path,self.name_list[idx]),gray=True)
        edge = imread(join(self.edge_path,self.name_list[idx]),gray=True) 
        #* ==================================================================


        depth_edge = imread(join(self.__depth_edge_path,self.name_list[idx]),gray=True)
        normal_map = imread(join(self.__normal_path,self.name_list[idx]))
        normal_edge  = imread(join(self.__normal_edge_path,self.name_list[idx]),gray=True)

        
        return image,depth,label,edge,depth_edge,normal_map,normal_edge,self.name_list[idx]
        
        

    


    
    def getitem_PIL(self,idx):

        image = imread_PIL(join(self.__image_path,self.name_list[idx]))
        depth = imread_PIL(join(self.__depth_path,self.name_list[idx]),gray=True)

        #* no cropped ========================================================
        label = imread_PIL(join(self.labels_path,self.name_list[idx]),gray=True)
        edge = imread_PIL(join(self.edge_path,self.name_list[idx]),gray=True)
        #* ==================================================================

        
        depth_edge = imread_PIL(join(self.__depth_edge_path,self.name_list[idx]),gray=True)
        normal_map = imread_PIL(join(self.__normal_path,self.name_list[idx]))
        normal_edge  = imread_PIL(join(self.__normal_edge_path,self.name_list[idx]),gray=True)
        
        return image,depth,label,edge,depth_edge,normal_map,normal_edge,self.name_list[idx]

        




def readtxt(path):
    return np.loadtxt(path,dtype=np.str0,delimiter='\n')


if __name__ == "__main__":
    Nyud2(gen_edge=True)
    # Nyud2(gen_edge=False)