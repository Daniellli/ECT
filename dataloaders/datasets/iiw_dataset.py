'''
Author: daniel
Date: 2023-02-08 17:28:27
LastEditTime: 2023-08-01 11:45:02
LastEditors: daniel
Description: 
FilePath: /Cerberus-main/dataloaders/datasets/iiw_dataset.py
have a nice day
'''
import torch
from PIL import Image
from os.path import exists, join, split
import numpy as np
import json
import os 
from os.path import join,split,exists

# import data_transforms as transforms

import utils.data_transforms as transforms
import cv2

from os.path import join, split, exists

class IIWDataset(torch.utils.data.Dataset):
    def __init__(self,data_transforms=None, data_dir = '/DATA2/cxx/IIW/IIW/IIW/iiw-dataset/',
                 split = 'train',out_name = True, guidance = True):
        self.data_dir = data_dir
        if data_transforms is None:
            info = json.load(open(join(data_dir, 'info.json'), 'r'))
            normalize = transforms.Normalize(mean=info['mean'],
                                            std=info['std'])
            crop_size = 512
            naive_t = [
                    transforms.Resize(crop_size),
                    transforms.ToTensorMultiHead(),
                    normalize]
            self.transforms =  transforms.Compose(naive_t)
        else :
            self.transforms = data_transforms

            
        self.image_list = None
        self.image_data_list = data_dir + '/data'
        self.split = split
        self.read_lists()
        self.out_name = out_name
        self.guidance = guidance
        self.guide_size = 512 #? 
        
        #* EDTER dataloader 
        # save_list = []
        # for name in self.image_list:
        #     save_list.append(' '.join([join('data',name+".png"),join('data',name+".mat")]))

        # np.savetxt(join(data_dir,'ImageSets/test.txt'),save_list,fmt='%s')

    
        
        
        
    def get_label(self,label_path):
        
        #darker_dict = {'E':0,'C':1,'W':2}
        label_data = json.load(open(label_path,'rb'))
        comparisons = label_data['intrinsic_comparisons']
        points = label_data['intrinsic_points']
        point_dict = {}
        for point in points:
            point_dict[point['id']] = [point['x'],point['y']]
        comparison_point = []  #* each element represent the position of point1 and point2 
        #? what is the  darder means?  
        comparison_label = []  #* each element represent  the darker and darker score in the pair between point1 and point2   
        for pair in comparisons:
            comparison_point.append([point_dict[pair['point1']],
                                    point_dict[pair['point2']]])
            comparison_label.append([pair['darker'],pair['darker_score']])
        return comparison_point, comparison_label      
    
    def name2idx(self,name):
        return self.image_list.index(name)
    
    def __getitem__(self, index):
        img_name = self.image_list[index] + '.png'
        label_name = self.image_list[index] + '.json'
        data = [Image.open(join(self.image_data_list, img_name))]

        #get label
        label_dir = join(self.image_data_list, label_name)
        points, labels = self.get_label(label_dir) #* the value range of point is between 0 and 1 

        data = list(self.transforms(data[0],np.array(points))) 
        return data[0],self.image_list[index]#* return the image and name only



    def getitem(self, index):
        img_name = self.image_list[index] + '.png'
        label_name = self.image_list[index] + '.json'
        data = [Image.open(join(self.image_data_list, img_name))]

        #* not gray situation 
        # data = np.array(data[0])
        # if len(data.shape) == 2: # gray scale
        #     data = np.stack([data , data , data] , axis = 2)
        # data = [Image.fromarray(data)]
        
        #get label
        label_dir = join(self.image_data_list, label_name)
        points, labels = self.get_label(label_dir) #* the value range of point is between 0 and 1 

        if self.guidance: #* ?
            # from IPython import embed
            # embed()
            guide_image = data[0].resize((self.guide_size, self.guide_size),Image.ANTIALIAS)
            #* visualize for debug
            # guide_image.save('q.jpg')
            guide_image = self.transforms.transforms[-2](guide_image)[0]
            guide_image = self.transforms.transforms[-1](guide_image)[0]
            # cv2.imwrite('v.jpg',guide_image.permute(1,2,0).numpy()*255)
            
        data = list(self.transforms(data[0],np.array(points))) 
        # data = list(self.transforms(data[0]),self.transforms(np.array(points))) 
        data.append(labels) 

        if self.out_name: #* whether output the image name, by default is true 
            data.append(self.image_list[index])

        if self.guidance: #* whether output the data transformed manually, by default is true
            data.append(guide_image)

        #*  [image, the point  corresponding to the label, the label corresponding to the points, image name, transformation guidance image ]
        
        return tuple(data) 
        # data = tuple(data) 
        # return data[0],data[3]#* return the image and name only
    
    def getitem_by_name(self, name):
        img_name = name + '.png'
        label_name = name + '.json'
        data = [Image.open(join(self.image_data_list, img_name))]
        
        #get label
        label_dir = join(self.image_data_list, label_name)
        points, labels = self.get_label(label_dir) #* the value range of point is between 0 and 1 
            
        data = list(self.transforms(data[0],np.array(points))) 
        # data = list(self.transforms(data[0]),self.transforms(np.array(points))) 
        data.append(labels) 

        #*  [image, the point  corresponding to the label, the label corresponding to the points, image name, transformation guidance image ]        
        return tuple(data) 
    
    
    def getitem_all(self, index):
        img_name = self.image_list[index] + '.png'
        label_name = self.image_list[index] + '.json'
        data = [Image.open(join(self.image_data_list, img_name))]

        #get label
        label_dir = join(self.image_data_list, label_name)
        points, labels = self.get_label(label_dir) #* the value range of point is between 0 and 1 

        # data = list(self.transforms(data[0],np.array(points))) 
        # return data[0],self.image_list[index]#* return the image and name only
        return data,points, labels


    def __len__(self):
        return len(self.image_list)

    def read_lists(self):
        image_path = join(self.data_dir, self.split+'.txt')
        assert exists(image_path)
        self.image_list = [line.strip() for line in open(image_path, 'r')]



if __name__ == '__main__':

    data_root= "/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/data/IIW/iiw-dataset"
    info = json.load(open(join(data_root, 'info.json'), 'r'))
    
    normalize = transforms.Normalize(mean=info['mean'],
                                     std=info['std'])
    crop_size = 512

    naive_t = [transforms.Resize(crop_size),
            transforms.ToTensorMultiHead(),
            normalize]
    
    trainset = IIWDataset(data_transforms=transforms.Compose(naive_t),data_dir=data_root)

    max_point_num = 0
    sample = trainset.__getitem__(0)
    # print(all_data)


