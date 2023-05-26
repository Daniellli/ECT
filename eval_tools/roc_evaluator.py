'''
Author: daniel
Date: 2023-05-24 19:12:23
LastEditTime: 2023-05-25 17:42:10
LastEditors: daniel
Description: 
FilePath: /Cerberus-main/eval_tools/roc_evaluator.py
have a nice day
'''

import os 

from os.path import join,split,exists
from dataloaders.prediction_loaders.rind_prediction_loader import *  


from dataloaders.datasets.bsds_hd5_test import MydatasetTest

from sklearn.metrics import precision_recall_curve, auc, roc_curve, roc_auc_score
from  tqdm import tqdm
import time

import matplotlib.pyplot as plt
class AUROCEvaluator:


    def __init__(self,prediction_data_loader,gt_loader):
        self.prediction_data_loader = prediction_data_loader
        self.gt_loader = gt_loader

        self.save_dir = join(self.prediction_data_loader.root,'auroc')
        if not exists(self.save_dir):
            os.makedirs(self.save_dir)
                 


    def __call__(self,label='A'):


        all_gts = []
        all_preds = []

        tic = time.time()

        
        for index in tqdm(range(self.gt_loader.__len__())):

            label = self.gt_loader.getitem_label(index)
            gt_edge = np.max(np.concatenate([ x[...,None] for x in label],axis = -1),axis=-1)
            # label.append(gt_edge)
            # show_imgs(label,[1]*len(label))

            #* prediction
            re_mns,ie_mns,ne_nms,de_nms, predicted_rind = self.prediction_data_loader.getitem_by_name(self.gt_loader.idx2name(index))
            # show_imgs([re_mns,ie_mns,ne_nms,de_nms,predicted_rind],[1,1,1,1,1])

            all_gts.append(gt_edge.reshape(-1))
            all_preds.append(predicted_rind.reshape(-1))


        fpr, tpr, _ = roc_curve(np.concatenate(all_gts), np.concatenate(all_preds))

        auroc_score_1 = auc(fpr, tpr,)
        print('AUROC is: ', auroc_score_1)
        print('FPR95 is: ', fpr[tpr > 0.95][0])
        print('spend time : ',time.strftime("%H:%M:%S",time.gmtime(time.time() - tic)))

        line = plt.plot(fpr, tpr)
        return line,auroc_score_1
        # plt.xlabel("FPR")
        # plt.ylabel("TPR")
        # plt.title("AUROC: " + "%.4f"%(auroc_score_1))
        # plt.savefig(join(self.save_dir,'AUROC.jpg'))
        
        
        # print('FPR95 is: ', fpr[tpr > 0.95][0])


if __name__ == "__main__":
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams['savefig.dpi'] = 500 #图片像素
    plt.rcParams['figure.dpi'] = 500 #分辨率
    
    fig = plt.figure(figsize=(5, 4))

    # loader = PredictionLoader('/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/networks/need2release/wo_containt_loss_0')
    EA2loader = PredictionLoader('/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/networks/need2release/wo_cause_interaction_0') #* use EA2 Loss
    
    NoEA2loader = PredictionLoader('/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/networks/need2release/without_cause_interaction_and_constraint_loss_0')
    
    test_dataset = MydatasetTest(root_path='/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/data/BSDS-RIND/BSDS-RIND-Edge/Augmentation')

    auroc_evaluator = AUROCEvaluator(EA2loader,test_dataset)
    l1,auroc1 = auroc_evaluator(label='l1')

    auroc_evaluator = AUROCEvaluator(NoEA2loader,test_dataset)
    l2,auroc2 = auroc_evaluator(label='l2')

    plt.legend(handles=[l1[0],l2[0]],labels=['w/ EA2 loss (AUROC: %.3f)'%(auroc1),'w/o EA2 loss (AUROC: %.3f)'%(auroc2)],loc='best')

    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    # plt.title("AUROC ")

    plt.xticks(np.linspace(0, 1, 3))
    

    plt.yticks(np.linspace(0, 1, 3))
    
    

    plt.savefig(join('logs','AUROC.jpg'))

