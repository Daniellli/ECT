from os.path import join,exists,split
import torch
import math

import json

import time
from utils import * 

import os 
from tqdm import tqdm
from loguru import logger

# os.chdir('/home/DISCOVER_summer2022/xusc/exp/Cerberus-main')
# from dataloaders.datasets.nyud_geonet import NYUD_GeoNet

from dataloaders.datasets.iiw_dataset import IIWDataset
import numpy as np 
import cv2
import scipy.io as scio
import argparse

from utils.utils import * 

from torchvision import transforms






def dump_json(path,value):
        with open (path,'w') as f:
            json.dump(value,f)


def load_json(path):
            with open(path,'r') as f :
                return json.load(f)

def normalize(data):
    return (data-data.min())/(data.max()-data.min())


def draw_arraw(src,p1,p2,color=(0,0,255),thickness=2,line_type=0,shift=0,tipLength=0.1):
    """
    # int thickness #线宽
    # int line_type #线的类型
    # int shift #箭头位置平移系数
    # double tipLength  #箭头大小缩放系数
    """
    # Mask = 255*np.ones((100,100,3), dtype=np.int)
    # Mask = np.array(Mask, dtype='uint8')
    cv2.arrowedLine(src,p1, p2, color,thickness,line_type,shift,tipLength)



class IIWEvaluator:

    def __init__(self,iiw_dataset,eval_dir) -> None:
        
        self.dataloader = iiw_dataset
        self.eval_dir = join(eval_dir,'reflectance','met')
        self.eval_root = eval_dir
        self.eval_json_root = join(eval_dir,'json')
        make_dir(self.eval_json_root)
        
        

    def eval_one_threshold(self,pred_xy,labels_points,labels,shape):
        
        
        def inside_square(axis_x,axis_y,pred_edge_xy):
             #* filter the square
            min_border = np.where(pred_edge_xy[:,0] >= axis_x.min(),1,0)
            max_border = np.where(pred_edge_xy[:,0] <= axis_x.max(),1,0)
            pred_edge_xy = pred_edge_xy[min_border == max_border]

            min_border = np.where(pred_edge_xy[:,1] >= axis_y.min(),1,0)
            max_border = np.where(pred_edge_xy[:,1] <= axis_y.max(),1,0)
            pred_edge_xy = pred_edge_xy[min_border == max_border]
            return pred_edge_xy

        def in_same_line(p1,p2,xy,error_threshold = 1):
            
            vector1=p1-xy 
            vector2=p2-xy 
            # if vector1[0]/(vector1[1]+1e-5)  - vector2[0]/(vector2[1]+1e-5) < 1e-4:
            #     pass
            # print(abs(vector1[0]*vector2[1] -vector2[0]*vector1[1] ) / ((p1-p2)**2).sum()**(0.5))
            if abs(vector1[0]*vector2[1] -vector2[0]*vector1[1] ) / ((p1-p2)**2).sum()**(0.5)  < error_threshold: #* 1e+1, 4e+1 and 5e+1  the bound inside the line,  
            # if : #* 1e+1, 4e+1 and 5e+1  the bound inside the line,  
                return True

            return False


        total_pair = 0
        recall_pair=0
        #!=============================
        #? why 
        pred_xy = np.concatenate([pred_xy[:,1][:,None],pred_xy[:,0][:,None]],axis=1)
        
        # for (x,y) in pred_xy:
        #     self.draw_point(self.tmp_img,(x,y),point_color=self.gen_color(),point_size=1)

        #!=============================
        for idx,(point,label)  in enumerate(zip(labels_points,labels)):
            if label[0] != 'E':
                total_pair+=1
                p1 = (point[0]*shape).int()
                p2 = (point[1]*shape).int()

                p12=torch.cat([p1[None,...],p2[None,...]],axis=0).numpy()
                axis_x = p12[:,0]
                axis_y = p12[:,1]

                pred_edge_xy= inside_square(axis_x,axis_y,pred_xy.copy())
                
                #* debug
                # self.tmp_img = self.draw_point_pair(self.tmp_img,p1.numpy(),p2.numpy())

                for idx in pred_edge_xy:
                    xy = torch.from_numpy(idx).int()
                    if  in_same_line(p1,p2,xy,1):
                        #* debug
                        # print('xy:',x,y,'\t p1:',p1,'\t p2:',p2)
                        # self.draw_point(self.tmp_img,xy.numpy(),point_color=self.gen_color(),point_size=1)
                        # self.draw_point(self.tmp_img,xy.numpy(),point_color=self.gen_color(),point_size=4)
                        recall_pair+=1
                        #todo draw the point 
                        break

        #* debug
        # cv2.imwrite(self.gen_name()+'.jpg',self.tmp_img)

        # recall = recall_pair/total_pair
        # print("recall : %.4f \t total_pair : %d \t recall pair : %d "%(recall,total_pair,recall_pair))
        return recall_pair/total_pair




        
    def draw_label(self,img,label_points,labels):
        if img.max()<=1:
            img = np.array(img*255,dtype=np.uint8)

        img = check_img(img)
        image_shape=torch.from_numpy(np.array(img.shape[:-1]))

        for idx,(points,label) in enumerate(zip(label_points,labels)):#* value range is between 0 and 1 

            # print(labels[idx])
            if label[0]!='E':
                p1= (points[0]*image_shape).int().numpy()
                p2= (points[1]*image_shape).int().numpy()
                self.draw_point_pair(img,p1,p2)
                
        self.draw_image(img)
        

    def draw_point(self,img,point,point_color=(255,0,0),point_size = 1,thickness = 4):
        # thickness = 4 # 可以为 0 、4、8
        cv2.circle(img,point, point_size, point_color, thickness)

    def gen_color(self):
        return (np.random.random((1,3))*255).astype(np.int32)[0].tolist() # BGR

    def draw_point_pair(self,img,p1,p2):
        img = check_img(img)
        point_color = self.gen_color()
        self.draw_point(img,p1,point_color)
        self.draw_point(img,p2,point_color)
        draw_arraw(img,p1,p2,color=point_color,thickness=2,tipLength=0.09)
        return img
        
    def gen_name(self):
        # return str(time.time()).replace('.','')
        return time.strftime("%Y-%m-%d-%H:%M:%s",time.gmtime(time.time()))

    def draw_image(self,im,mode=0):
        # show_imgs([im],[mode])
        show_imgs([im],[mode],img_name=self.gen_name()+'.jpg')

            
    '''
    description:  check whether there exist  the pair not equal to 'E'
    param {*} self
    param {*} labels
    return {*}
    '''
    def check_labels(self,labels):
        num = (np.array(labels)[:,0] != 'E').sum()
        if num != 0 :
            return True
        else :
            return False


        
    def eval_one_image(self,idx):
    
        image,labels_points,labels,name,guidance =  self.dataloader.getitem(idx)
        if not self.check_labels(labels=labels):
            #* not positive pair 
            dump_json(join(self.eval_json_root,name+".json"),{})
            print(f"there is not positive pair in {name}.json")
            return 

        
        inference_res = self.load_mat(join(self.eval_dir,name+'.mat'))

        #* threshold  from 0.01 to 0.99
        # thresholds=np.arange(0.5,1,0.01)
        thresholds=np.arange(0.01,1,0.01)
        # thresholds=[0.5]
        recall_dict = {}

        # self.tmp_img = np.array(transforms.ToPILImage()(normalize(image)))
        # imwrite('origin_image.jpg',self.tmp_img)
        
        for threshold in thresholds:
            
            pred_edge_xy =  np.argwhere(inference_res > threshold)#* 1 if the response value large than threshold else 0
            
            #* debug 
            # imwrite('edge_with_%d_threshold.jpg'%(threshold*100),np.array((inference_res > threshold)*255,dtype=np.uint8))
            # print(threshold,pred_edge_xy.shape)
            
            recall = self.eval_one_threshold(pred_edge_xy,labels_points,labels,np.array(inference_res.shape))

            # recall_list.append(recall)
            recall_dict[threshold]=recall


        dump_json(join(self.eval_json_root,name+".json"),recall_dict)
        

    def eval(self):
        
        #* calculate recall for each image 
        tic = time.time()
        process_mp(self.eval_one_image,range(self.dataloader.__len__()),num_threads=256)
        # for idx in tqdm(range(self.dataloader.__len__())):
        #     self.eval_one_image(idx)
        #* debug 
        # self.eval_one_image(10)
        print('spend time : ',time.strftime("%H:%M:%S",time.gmtime(time.time()- tic)))

        #* compute the mean recall 
        eval_dict={}
        for name in self.dataloader.image_list:
            data = load_json(join(self.eval_json_root,name+'.json'))
            if len(data) == 0 :
                print(name,'is None')
                continue
            eval_dict[name] = np.array(list(data.values())).mean()
     

        m_recall = np.array(list(eval_dict.values())).mean()

        print(f'mean recall : {m_recall}')
        dump_json(join(self.eval_root ,'eval.json'),eval_dict)
        dump_json(join(self.eval_root ,'eval_res.json'),{'mean_recall':m_recall})
        

        return m_recall


    # def load_mat(self,path):
    #     data = scio.loadmat(path)
    #     return data['groundTruth'][0,0]['Boundaries'][0,0]

    def load_mat(self,path):
        data = scio.loadmat(path)
        return data['result']



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d','--eval-data-dir', 
        default='/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/networks/need2release/iiw_1',
        help="eval data dir, must be absolution dir ")

    args = parser.parse_args()
    data_root = '/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/data/IIW/iiw-dataset'
    evaluator = IIWEvaluator(IIWDataset(data_dir=data_root,split='test'),args.eval_data_dir)
    evaluator.eval()