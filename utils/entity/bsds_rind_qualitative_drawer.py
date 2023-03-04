






from utils.entity.bsds_rind_gt_loader import * 

import numpy as np 

from utils import * 
from plot.plot_utils import * 

class QualitativeDrawer:

    def __init__(self,root,name,save_dir = None ,gt_loader =None):


        if gt_loader is None :
            self.gt_loader = GtLoader()

        if save_dir is None : 
            self.save_dir = join('logs/bsds',name)
        else:
            self.save_dir = save_dir

        make_dir(self.save_dir)


        self.root = root
    
        self.name  = name

        self.name_list = np.loadtxt('plot/bsds_eval_list.txt',dtype=np.str0)

        self.RINDE_COLOR = [
        (10,139,226),
        (142,217,199),
        (235,191,114),
        (174, 125, 176),
        (219, 118, 2)
        ]


        self.COLORS = {
            # "TP":(10,251,9),
            # "TP":(20,251,9),
            # "TP":(100,251,9),
            "TP":(20,255,10),
            "FN":(255,10,10),#* 蓝色
            # "FP":(7,255,252),
            "FP":(251,254,3),
            # "FN":(255,255,0),#* 青色
            # "FN":(255,0,255),#* 深红色
            # "FN":(203,192,255),#* 粉色
        }
    
        #* load performance of each task for each image 
        self.TASKS = ["reflectance","illumination","normal","depth","all_edges"]
        self.illumination  =self.get_images_quantitative_results('illumination')
        self.normal  =self.get_images_quantitative_results('normal')
        self.reflectance  =self.get_images_quantitative_results('reflectance')
        self.depth  =self.get_images_quantitative_results('depth')
    
    def __len__(self):
        return  len(self.name_list)
    
    def get_name(self):
        return self.name

    def get_images_quantitative_results(self,task):
        
        assert task in self.TASKS
        # return np.loadtxt(join(self.root,task,'nms-eval','eval_bdry_img.txt'),dtype=np.str0)[idx]
        return np.loadtxt(join(self.root,task,'nms-eval','eval_bdry_img.txt'))

    
    def getitem_by_task(self,task,idx):
        name = self.name_list[idx]
        image  = self.gt_loader.get_image(name)

        prediction_edge  = imread(join(self.root,task,'nms',self.name_list[idx]+'.png'),gray=True)

        return image,prediction_edge,name


    def get_quantitatives(self,index):
        
        name = self.name_list[index]
        reflectance =self.reflectance[index]
        illumination = self.illumination[index]
        normal =self.normal[index]
        depth = self.depth[index]

        return reflectance,illumination,normal,depth,name
    
    def get_F_quantitatives(self,index):
        name = self.name_list[index]
        reflectance =self.reflectance[index][-1]
        illumination = self.illumination[index][-1]
        normal =self.normal[index][-1]
        depth = self.depth[index][-1]
        return reflectance,illumination,normal,depth,name

    
        
    def get_task_F_quantitatives(self,task,index):
        return getattr(self,task)[index][-1],self.name_list[index]

    def name2idx(self,name):
        
        return np.where(self.name_list == name)[0][0]

    def draw_edge(self,task,idx,threshold_range = np.arange(0.5,1,0.1)):


        image,prediction_edge,name = self.getitem_by_task(task,idx)#* 0-255 map 
        
        name_format = "%s_%d.png"
        save_dir = join(self.save_dir,task)
        make_dir(save_dir)

        gt_edge = self.gt_loader.get_edge_by_task(task,self.name_list[idx])#* 0-1 map 

        
        ans=[]
        for threshold in threshold_range:

            
            save_name = join(save_dir, name_format%(name,threshold*100))
            
            # if exists(save_name):
            #     return [imread(save_name)[:,:,::-1]]

            image_drawed = image.copy()[:,:,::-1]

            filter_pred_edge = (prediction_edge>threshold*255)

            #* FP
            image_drawed [(gt_edge==0) & ( filter_pred_edge ==1)] = self.COLORS['FP']

            #* FN
            image_drawed[(gt_edge==1)  &  (filter_pred_edge ==0)] = self.COLORS['FN']

            # print(f'before {(filter_pred_edge==1).sum()}')
            filter_pred_edge = dilation(filter_pred_edge,times=1.2)/255
            # print(f'after {(filter_pred_edge==1).sum()}')
            #* TP
            image_drawed[(gt_edge == 1 ) &  (filter_pred_edge==1)] = self.COLORS['TP']

            show_imgs([image_drawed],[0],img_name=save_name)
            

            ans.append(image_drawed)


        return ans
            

     
    



if __name__ ==  "__main__":

    RINDNET_ROOT="/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/networks/trash/precomputed"
    our_drawer = QualitativeDrawer('/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/networks/need2release/full_version_0','ours')
    our_drawer.draw_edge('normal',0)
    our_drawer.get_task_F_quantitatives('normal',0)
    our_drawer.get_F_quantitatives(0)
    
  


    