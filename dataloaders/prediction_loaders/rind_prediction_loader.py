





from dataloaders.prediction_loaders.base_loader import *


import json 
from IPython import embed
    



class ReflectanceLoader(BaseLoader):
    def __init__(self, path):
        super(ReflectanceLoader, self).__init__(path)

    
class IlluminationLoader(BaseLoader):
    def __init__(self, path):
        super(IlluminationLoader, self).__init__(path)
        


class NormalLoader(BaseLoader):
    def __init__(self, path):
        super(NormalLoader, self).__init__(path)
        

class DepthLoader(BaseLoader):
    def __init__(self, path):
        super(DepthLoader, self).__init__(path)

    


class PredictionLoader:


    def __init__(self, path):

        self.root = path 
        
        self.RL = ReflectanceLoader(join(path, 'reflectance'))
        self.IL = IlluminationLoader(join(path, 'illumination'))
        self.NL = NormalLoader(join(path, 'normal'))
        self.DL = DepthLoader(join(path, 'depth'))


        with open(join(path,'eval_res.json'),'r') as f :
            self.matrics = json.load(f)

        for k,v in self.matrics.items():
            print(k,v)
        
        
        
    def __len__(self):
        return len(self.RL.__len__())

    
    def name2idx(self,name):
        return self.IL.name2idx(name)


    def getitem_by_name(self,name):
        return self.getitem(self.name2idx(name))

    def getitem(self,idx):
        re,re_mns = self.RL.getitem(idx)
        ie,ie_mns = self.IL.getitem(idx)
        ne,ne_nms = self.NL.getitem(idx)
        de,de_nms = self.DL.getitem(idx)

        return re_mns,ie_mns,ne_nms,de_nms,np.max(np.concatenate([re_mns[...,None],ie_mns[...,None],ne_nms[...,None],de_nms[...,None]],-1),axis=-1)
        
        


if __name__ == "__main__":

    loader = PredictionLoader('/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/networks/need2release/wo_containt_loss_0')
    loader.getitem(0)

    

    

        
        

        




    