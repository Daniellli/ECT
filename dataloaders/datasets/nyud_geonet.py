

import os
import cv2
import numpy as np
import torch.utils.data as data
import scipy.io as sio

from concurrent import futures
from six.moves import urllib
from natsort import natsorted
from plot.plot_main_pic import dilation
import cv2
import matplotlib.pyplot as plt
import os.path as osp 
import scipy.io as scio
import torch
import torch.nn.functional as F
from utils import * 
import torchvision.transforms as transforms




class NYUD_GeoNet(data.Dataset):
    """
    NYUD training dataset collected from raw NYUD dataset, provided by GeoNet
    depth and normal

    Data can also be found at:
    % Depth Intrinsic Parameters
    fx_d = 5.8262448167737955e+02;
    fy_d = 5.8269103270988637e+02;
    cx_d = 3.1304475870804731e+02;
    cy_d = 2.3844389626620386e+02;

    """

    def __init__(self,
                 root='/home/aidrive/zyh/yuhang/dataset/nyu2d/',
                 transform=None,
                 retname=True,
                 split='train',
                 depth_mask=False,
                 generation_gt = False,
                 mode='test'
                 ):

        self.root = root
        
        self.mode= mode
        
        self.cam_intrinsic = np.float32(np.array([[582.64, 0, 313.04],
                                                  [0, 582.69, 238.44],
                                                  [0, 0, 1]]))

        self.retname = retname
        self.split = split
        self.depth_mask = depth_mask

        # Original samples: contain image, depth, normal, valid_masks
        self.sample_ids = []
        
        self.sample_dir = os.path.join(root, self.split) 

        if self.split == 'train':
            self.sample_ids = natsorted(os.listdir(self.sample_dir))
            self.samples = [os.path.join(self.sample_dir, x) for x in self.sample_ids]
        else:
            self.sample_ids = natsorted(os.listdir(os.path.join(self.sample_dir, 'images')))

        # Display stats
        print('Number of dataset images: {:d}'.format(len(self.sample_ids)))
        if transform is not None:
            self.transform = transform
        elif mode == 'inference':
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            self.transform = transforms.Compose([
                                transforms.ToTensor(),
                                normalize])  ## pre-process of pre-trained model of pytorch resnet-50
        else:

            self.transform = None
        
        #!============================================================================
        #* only for val subset 
        self.edge_gt_path=os.path.join(self.sample_dir,'edge_gt')
        self.edge_gt_list= natsorted(os.listdir(self.edge_gt_path))


        DEPTH_THRESHOLD=0.01
        NORMALS_THRESHOLD=0.03

        self.depth_path = os.path.join(self.sample_dir,'depth_%s'%(str(DEPTH_THRESHOLD).replace('.','@')))
        self.depth_mat_path = os.path.join(self.sample_dir,'depth_%s_mat'%(str(DEPTH_THRESHOLD).replace('.','@')))
        make_dir(self.depth_path)
        make_dir(self.depth_mat_path)
        self.depth_list = []

        self.normal_path = os.path.join(self.sample_dir,'normal_%s'%(str(NORMALS_THRESHOLD).replace('.','@')))
        self.normal_mat_path = os.path.join(self.sample_dir,'normal_%s_mat'%(str(NORMALS_THRESHOLD).replace('.','@')))
        make_dir(self.normal_path)
        make_dir(self.normal_mat_path)
        self.normal_list = []
        
        
        if generation_gt:
            self.gt_map_generation(num_threads=256)

        # self.label_path= '/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/data/nyud2/NYU_origin/nyu_labels'
        # self.label_list = os.listdir(self.label_path)

        #!============================================================================
       

    def __len__(self):
        return len(self.sample_ids)

    def __str__(self):
        return 'NYUD_GeoNet dataset for depth and normal estimation.'


    def _get_normal_depth_edge(self,data):
        sample,name = data

        depth_mat=osp.join(self.depth_mat_path,name+".mat") 
        depth_img = osp.join(self.depth_path,name+".png")
        normal_mat=osp.join(self.normal_mat_path,name+".mat")
        normal_img=osp.join(self.normal_path,name+".png")

        if not osp.exists(normal_img) or not  osp.exists(depth_img)  or not   osp.exists(depth_mat) or not   osp.exists(normal_mat) :
            depth_edge,normal_edge = get_normal_depth_edge(sample["depth"],sample["normals"])
            #!=====================
            #* edge map to filter further 
            edge_filter=False
            if edge_filter :
                new_edge=dilation(sample['edge'],5)
                new_edge = F.interpolate(torch.from_numpy(new_edge)[None,None],size=depth_edge.shape)
                new_edge=new_edge.squeeze().numpy()
                depth_edge[new_edge!=255] =0
                normal_edge[new_edge!=255] =0
            #!=====================
            self.save_as_mat(depth_mat,depth_edge)
            self.save_as_mat(normal_mat,normal_edge)
            cv2.imwrite(depth_img,depth_edge*255)
            cv2.imwrite(normal_img,normal_edge*255)
    
    def gt_map_generation(self,num_threads=64):
        num_sample = self.__len__()
        parameter_list = [self.__getitem__(idx) for idx in range(num_sample)]
        process_mp(self._get_normal_depth_edge,parameter_list,num_threads=num_threads)
        

    


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



    

    def get_one_sample(self, index):
        sample = {}
        if self.split == 'train':
            _img, _depth, _norm  = self._load_training_sample(index)
        elif self.split == 'val':
            _img, _depth, _norm = self._load_val_sample(index)
            # label = cv2.imread(osp.join(self.label_path,self.label_list[index]))
            # label[label>0] ==1
            # sample['label'] = label 
            
            _edge = self.load_edge(index)
            sample['edge'] = _edge    
            

        #* for inference
        sample['image'] = _img
        sample['original_image'] = _img
        sample['depth'] = _depth
        sample['normals'] = _norm
        sample['intrinsic'] = self.cam_intrinsic

        if self.retname:
            sample['meta'] = {'image': str(self.sample_ids[index]),
                              'im_size': (_img.shape[0], _img.shape[1])}
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    
    def get_image(self, index):
        _img, _depth, _norm = self._load_val_sample(index)
        _img= _img

        if self.transform is not None:
            _img = self.transform(_img)

        return _img

    def __getitem__(self, index):
        
        name = '.'.join(self.sample_ids[index].split('.')[:-1])
        if self.split == 'val' or self.split == 'test':
            if self.mode != 'inference':
                sample = self.get_one_sample(index)  # if use try except; when iterate the object, cannot stop for loop
            else:
                sample = self.get_image(index)  # if use try except; when iterate the object, cannot stop for loop

        else:
            flag = True
            while flag:
                try:
                    sample = self.get_one_sample(index)
                    flag = False
                except:
                    tmp = np.random.randint(0, self.__len__(), 1)[0]
                    print("data load error!", index, "use:  ", tmp)
                    index = tmp

        
        return sample,name

    def _load_training_sample(self, index):

        """
        Each sample are padded by one pixel (right and bottom) to have size 481 x 641
        fields:
        depth: callibrated depth maps, "0" indicates invalid measurements which you should ignore when training model
        grid: used by our code
        img: if you want to convert the img to the original format you should do the folowing:
            img(:,:,1) = img(:,:,1) + 2* 122.175
            img(:,:,2) = img(:,:,2) + 2* 116.169
            img(:,:,3) = img(:,:,3) + 2* 103.508
            img        = uint8(img)
        mask: valid values for surface normal
        norm: precomputed surface normal following https://github.com/aayushbansal/MarrRevisited
        """

        sample = sio.loadmat(self.samples[index])
        img = sample['img'][:-1, :-1, :]  # [480, 640, 3]
        depth = sample['depth'][:-1, :-1]  # [480, 640]
        norm = sample['norm'][:-1, :-1, :]  # [480, 640, 3]
        mask = sample['mask'][:-1, :-1]  # [480, 640]
        mask = mask > 0.5  # convert to boolean

        # convert to original format
        img[:, :, 0] = img[:, :, 0] + 2 * 122.175
        img[:, :, 1] = img[:, :, 1] + 2 * 116.169
        img[:, :, 2] = img[:, :, 2] + 2 * 103.508
        img = np.uint8(img)
        
        ####   (depth<0.1).sum() (depth==255).sum() (mask<0.5).sum()
                # invalid areas are set to 0
        norm[:, :, 0][~mask] = 0
        norm[:, :, 1][~mask] = 0
        norm[:, :, 2][~mask] = 0
        
#         depth[~mask] = 0

        return img, depth, norm


    def load_mat(self,path):
        return scio.loadmat(path)

    def load_edge(self,index):
        edge = self.load_mat(osp.join(self.edge_gt_path,self.edge_gt_list[index]))
        # edge['groundTruth'][0,0]['Segmentation'][0,0]
        # edge['groundTruth'][0,0]['Boundaries'][0,0]
        # edge['groundTruth'][0,0]['SegmentationClass'][0,0]

        return edge['groundTruth'][0,0]['Boundaries'][0,0]



    def _load_val_sample(self, index):
        img = np.load(os.path.join(self.sample_dir, 'images', self.sample_ids[index]))
        depth = np.load(os.path.join(self.sample_dir, 'depths', self.sample_ids[index]))
 
        #*  no use
        depth[depth>10.0] = 0
        _invalid_mask = np.ones_like(depth)
        _invalid_mask[45:471, 41:601] = 0
        _invalid_mask = _invalid_mask>0
        depth[_invalid_mask] = 0
        
        try:
            norm = np.load(os.path.join(self.sample_dir, 'normals', self.sample_ids[index]))

            # mask = np.load(os.path.join(self.sample_dir, 'masks', self.sample_ids[index]))
            # mask = mask > 0.5
            # norm[:, :, 0][~mask] = 0
            # norm[:, :, 1][~mask] = 0
            # norm[:, :, 2][~mask] = 0
            # if self.depth_mask:
            #     depth[~mask] = 0
            
        except:
            h,w = np.shape(depth)
            norm = np.zeros((h,w,3))
        
        
        # return img[45:471, 41:601,:], depth[45:471, 41:601], norm[45:471, 41:601,:] #* 426, 560
        # return img[46:471, 41:601,:], depth[46:471, 41:601], norm[46:471, 41:601,:]
        # return img[45:470, 41:601,:], depth[45:470, 41:601], norm[45:470, 41:601,:]
        return img,depth,norm
        
        
        







    
    


if __name__ == "__main__":
    db = NYUD_GeoNet(split='val',root='/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/data/nyud2',generation_gt=True)


    # exit()
    #* initial
    plt.rcParams['figure.figsize'] = (8.0, 4.0) # 设置figure_size尺寸
    plt.rcParams['savefig.dpi'] = 250 #图片像素
    plt.rcParams['figure.dpi'] = 250 #分辨率

    print(len(db))
    for i, (sample,name) in enumerate(db):
        
        # show_imgs([sample['depth'],sample['normals'],sample['image'],sample['edge']],[0,0,0,0])
        # show_imgs([sample['depth'],sample['normals'],sample['image'],sample['edge'],
        #         cv2.addWeighted(sample['image'],0.4,
        #         cv2.cvtColor(sample['edge']*255,cv2.COLOR_GRAY2RGB),0.6,0)],[0,0,0,0,0])
        
        # sample['edge']
        # depth_edge = cv2.imread(osp.join(db.depth_path,name+".png"),cv2.IMREAD_GRAYSCALE)
        # depth_merge= merge_images((normalize(sample['depth'])*255).astype(np.uint8),
        #                         0.5,sample['edge']*255,0.5)

        # _origin= merge_images(sample['image'],0.5,
        #                 np.concatenate([sample['edge'][...,None],sample['edge'][...,None],sample['edge'][...,None]],-1)*255,0.5)
        

        
        # cv2.imwrite('%06d_depth.png'%(i),np.concatenate([cv2.cvtColor(depth_merge,cv2.COLOR_GRAY2BGR),_origin,aaaa_origin],1))


        # normal_edge = cv2.imread(osp.join(db.normal_path,name+".png"),cv2.IMREAD_GRAYSCALE)
        
        # normal_merge= merge_images(
        #                             (normal_reverse_process(sample['normals'])*255).astype(np.uint8),0.5,
        #                             sample['edge']*255,0.5)
        # cv2.imwrite('%06d_normal.png'%(i),np.concatenate([normal_merge,_origin],1))
        print(i,name)

        
        if i == 5:
            break
