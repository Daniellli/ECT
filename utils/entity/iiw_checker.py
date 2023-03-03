
import numpy as np 
import cv2

import os 

from skimage import img_as_ubyte
from dataloaders.datasets.iiw_dataset import IIWDataset
import torch
from os.path import join, split, exists, isdir, isfile
import torchvision.transforms as transforms 
from utils import * 


def draw_point(img,point,point_color=(255,0,0),point_size = 1,thickness = 4):
    # thickness = 4 # 可以为 0 、4、8
    cv2.circle(img,point, point_size, point_color, thickness)

def gen_color():
    return (np.random.random((1,3))*255).astype(np.int32)[0].tolist() # BGR

def draw_point_pair(img,p1,p2,point_color = None ):
    img = check_img(img)

    if point_color is None :
        point_color = gen_color()

    
    draw_point(img,p1,point_color)
    draw_point(img,p2,point_color)
    draw_arraw(img,p1,p2,color=point_color,thickness=2,tipLength=0.09)


    return img
    


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


def draw_label(img,points,labels):
    if img.max()<=1:
        img = np.array(img*255,dtype=np.uint8)

    img = check_img(img)
    image_shape=torch.from_numpy(np.array(img.shape[:-1]))

    for idx,(points,label) in enumerate(zip(points,labels)):#* value range is between 0 and 1 

        # print(labels[idx])
        if label[0]!='E':
            p1= (points[0]*image_shape).int().numpy()
            p2= (points[1]*image_shape).int().numpy()
            img = draw_point_pair(img,p1,p2)
            
    return img



def draw_all_pairs(image,points,labels,shape=None):

 
    COLOR={
            "TP":(20,255,10),
            "FN":(255,10,10)
            # "FN":(10,10,255)
        }

    if shape is None:
        shape = np.array(image.shape[:-1])
        print(f'shape == {shape}')

    for idx,(point,label)  in enumerate(zip(points,labels)):

        if label[0] != 'E':
            p1 = (point[0]*shape).int()
            p2 = (point[1]*shape).int()
            
            image = draw_point_pair(image,p1.numpy(),p2.numpy(),COLOR['TP'])
    
    return image


def draw_activated_points(image,edge_map,points,labels,shape=(512,512)):
    
    
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
    
    #!=============================
    #? why 
    edge_map = np.concatenate([edge_map[:,1][:,None],edge_map[:,0][:,None]],axis=1)
    total_pair = recall_pair = 0
    
    COLOR={
            "TP":(20,255,10),
            "FN":(255,10,10)
            # "FN":(10,10,255)
        }

    #!=============================
    for idx,(point,label)  in enumerate(zip(points,labels)):
        if label[0] != 'E':
            total_pair+=1
            p1 = (point[0]*shape).int()
            p2 = (point[1]*shape).int()

            p12=torch.cat([p1[None,...],p2[None,...]],axis=0).numpy()
            axis_x = p12[:,0]
            axis_y = p12[:,1]

            pred_edge_xy= inside_square(axis_x,axis_y,edge_map.copy())
            
            #* debug
            
            recalled = False
            for idx in pred_edge_xy:
                xy = torch.from_numpy(idx).int()
                if  in_same_line(p1,p2,xy,1):
                    #* debug
                    # print('xy:',x,y,'\t p1:',p1,'\t p2:',p2)
                    # draw_point(image,xy.numpy(),point_color=gen_color(),point_size=1)
                    # draw_point(image,xy.numpy(),point_color=gen_color(),point_size=3)
                    # draw_point(image,xy.numpy(),point_color=gen_color(),point_size=5)
                    # draw_point(image,xy.numpy(),point_color=COLOR['TP'],point_size=3)
                    recall_pair+=1
                    recalled =True
                    break
            if recalled:
                image = draw_point_pair(image,p1.numpy(),p2.numpy(),COLOR['TP'])
            else:
                image = draw_point_pair(image,p1.numpy(),p2.numpy(),COLOR['FN'])


                    
        

    #* debug
    
    cv2.putText(image,f'{recall_pair} / {total_pair}', (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
    
    # cv2.imwrite(self.gen_name()+'.jpg',self.tmp_img)
    return image



class Checker:


    def __init__(self,root,save_path,iiw_dataset = None,):

        self.dataset = iiw_dataset

        self.to_imger =transforms.ToPILImage()
        self.root  = root
        
        self.pre_path = join(root,'reflectance','met')
        self.predictions = sorted(os.listdir(self.pre_path))
        
        self.debug_save_path = save_path
        if not exists(self.debug_save_path):
            make_dir(self.debug_save_path)

    def __len__(self):
        
        return len(self.predictions)
    
    def name2idx(self,name):
        return self.predictions.index(name)


    
    def getitem(self,index):
        
        pred = load_mat(join(self.pre_path , self.predictions[index]))['result']
        name = self.get_name(index)

        image,points, labels = self.dataset.getitem_by_name(name)

        return np.array(self.to_imger(normalize(image))),points, labels,img_as_ubyte(pred),name
    
    def get_name(self,index):
        return self.predictions[index].split('.')[0]
    

    
    def get_qualitative_result(self,index,thresholds_range = np.arange(0.5,1,0.1)):

        name = self.get_name(index)
        name_format = '%s_%d.jpg'
        image_dict = {}
        for threshold in thresholds_range:
            save_path = join(self.debug_save_path,name_format%(name,round(threshold*100,2)))
            image_dict[threshold] = imread(save_path)[:,:,::-1]
        return image_dict
    
    def draw_qualitative_results(self,index,thresholds_range = np.arange(0.5,1,0.1)):

        image,points, labels,pred, name = self.getitem(index)

        name_format = '%s_%d.jpg'
        
        for threshold in thresholds_range:
            save_path = join(self.debug_save_path,name_format%(name,round(threshold*100,2)))
            
            if not exists(save_path):
                pred_xys = np.argwhere(pred/255 > threshold)#* trheshold == 0.5
                draw_activated_points_image =draw_activated_points(image,pred_xys,points,labels,np.array(pred.shape))

                # show_imgs([(pred/255 > threshold).astype(np.uint8)*255,draw_activated_points_image],
                #             [1,0],['Prediction','Groundtruth'],
                #             img_name = save_path)

                show_imgs([draw_activated_points_image],
                            [0],img_name = save_path)
                
        
    
    


if __name__ == "__main__":
    

    run_iiw = "/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/networks/need2release/run_iiw"
    dataset = IIWDataset(data_dir='/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/data/IIW/iiw-dataset',split='test')

    rindnet_checker = Checker(join(run_iiw,'rindnet'),'logs/iiw_results_analysis/rindnet',dataset)
    hed_checker = Checker(join(run_iiw,'hed'),'logs/iiw_results_analysis/hed',dataset)
    our_checker = Checker('/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/networks/need2release/iiw_1',
                        'logs/iiw_results_analysis/ours',dataset)

        