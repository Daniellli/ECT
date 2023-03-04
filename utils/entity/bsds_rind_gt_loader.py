




from utils import *

from os.path import join,split, exists
class GtLoader:


    def __init__(self):

        self.origin_image_path = '/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/data/BSDS-RIND/test'
        self.gt_path = '/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/data/BSDS-RIND/testgt'


    ''' 
    description:  all edge value range is between 0 - 1 
    param {*} self
    param {*} name
    return {*}
    '''
    def getitem(self,name):
        image = imread(join(self.origin_image_path,name+'.jpg'))
        reflectance_edge = load_mat(join(self.gt_path,'reflectance',name+'.mat'))['groundTruth'][0,0]['Boundaries'][0,0]
        illumination_edge = load_mat(join(self.gt_path,'illumination',name+'.mat'))['groundTruth'][0,0]['Boundaries'][0,0]
        normal_edge = load_mat(join(self.gt_path,'normal',name+'.mat'))['groundTruth'][0,0]['Boundaries'][0,0]
        depth_edge = load_mat(join(self.gt_path,'depth',name+'.mat'))['groundTruth'][0,0]['Boundaries'][0,0]
        
        generic_edge = reflectance_edge | illumination_edge | normal_edge | depth_edge 


        return image,generic_edge,reflectance_edge,illumination_edge,normal_edge,depth_edge

    def get_image(self,name):
        # if exits(join(self.origin_image_path,name+'.jpg')):
        return imread(join(self.origin_image_path,name+'.jpg'))
            

    def get_rind_edge(self,name):
        reflectance_edge = load_mat(join(self.gt_path,'reflectance',name+'.mat'))['groundTruth'][0,0]['Boundaries'][0,0]
        illumination_edge = load_mat(join(self.gt_path,'illumination',name+'.mat'))['groundTruth'][0,0]['Boundaries'][0,0]
        normal_edge = load_mat(join(self.gt_path,'normal',name+'.mat'))['groundTruth'][0,0]['Boundaries'][0,0]
        depth_edge = load_mat(join(self.gt_path,'depth',name+'.mat'))['groundTruth'][0,0]['Boundaries'][0,0]
        return reflectance_edge,illumination_edge,normal_edge,depth_edge

    def get_edge_by_task(self,task,name):
        return load_mat(join(self.gt_path,task,name+'.mat'))['groundTruth'][0,0]['Boundaries'][0,0]





if __name__ ==  "__main__":
    loader = GtLoader()
    loader.get_image('2018')
    loader.getitem('2018')
    
    loader.get_edge_by_task('normal','2018')
    loader.get_rind_edge('2018')


    