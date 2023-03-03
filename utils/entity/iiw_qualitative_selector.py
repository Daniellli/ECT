


from utils import * 
import os 

from utils.entity.iiw_checker import *
import json 

from dataloaders.datasets.iiw_dataset import IIWDataset

class QulitativeSelector:

    def __init__(self,path,iiw_dataset,dataset_name = 'ours'):
        log_root = 'logs/iiw_results_analysis'
        
        self.root = path 
        self.log_path = join(log_root,dataset_name)
        
        self.json_path = join(path,'json')
        self.json_files  = sorted(os.listdir(self.json_path))
        self.checker = Checker(path,self.log_path,iiw_dataset)



    
    def __len__(self):
        return len(self.json_files )
    

    def get_quantitative_results_for_images(self):
        mean_metric_dict  = {}
        for idx,file_name in enumerate(self.json_files):
            with open(join(self.json_path,file_name),'r')as f :
                data = json.load(f)
            
            mean_metric_dict[file_name] = np.array(list(data.values())).mean()
        return mean_metric_dict
    
    def get_topx_idx(self,x):

        
        mean_metric_dict = self.get_quantitative_results_for_images()

        mean_metric_dict_after_sort = sorted(mean_metric_dict.items(), key =lambda k :k[1])
        
        # return [pair[0] for pair in mean_metric_dict_after_sort[::-1]][:x]
        return mean_metric_dict_after_sort[::-1][:x]


    

    def get_qualitatve_results(self,name,threshold_pickup = 0.5):
        idx = self.checker.name2idx(name)
        self.checker.draw_qualitative_results(idx,[threshold_pickup])
        image_list = self.checker.get_qualitative_result(idx,[threshold_pickup])
        return list(image_list.values())


    
def get_best_idx(main_dict,other_dict_list):

    
    n = len(other_dict_list)
    comparison_results = {}
    for main_k,main_value in main_dict.items():

        distance = 0 

        for d in other_dict_list:
            #* assume our is best 
            
            distance += (main_value - d[main_k])

        #* large is better
        comparison_results[main_k ] = distance/n
    


    return  sorted(comparison_results.items(), key =lambda k : k[1])[::-1]

        



if __name__ == "__main__":

    dataset = IIWDataset(data_dir='/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/data/IIW/iiw-dataset',split='test')


    ours_path = '/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/networks/need2release/iiw_1'
    rindnet_root = '/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/networks/need2release/run_iiw'
            
    ours = QulitativeSelector(ours_path,dataset,dataset_name='ours')
    hed = QulitativeSelector(join(rindnet_root,'hed'),dataset,dataset_name = 'hed')
    dff = QulitativeSelector(join(rindnet_root,'dff'),dataset,dataset_name = 'dff')
    rcf = QulitativeSelector(join(rindnet_root,'rcf'),dataset,dataset_name = 'rcf')
    rindnet = QulitativeSelector(join(rindnet_root,'rindnet'),dataset,dataset_name = 'rindnet')





