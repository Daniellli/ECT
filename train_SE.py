import os
import time
from cv2 import threshold
import numpy as np
import sys
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable


import copy
from model.edge_model import EdgeCerberus
import os.path as osp

from os.path import join,split, isdir,isfile, exists

import random
import wandb

from loguru import logger
from utils.loss import SegmentationLosses
from utils.edge_loss2 import AttentionLoss2
from dataloaders.datasets.bsds_hd5 import Mydataset
from torch.utils.data.distributed import DistributedSampler
from utils import save_checkpoint,AverageMeter,calculate_param_num



from utils.semantic_edge_option import parse_args

from utils.utils import *


import json
import warnings
warnings.filterwarnings('ignore')

import torch.distributed as dist
import torch.multiprocessing as mp
import json

from utils.global_var import *


from utils.check_model_consistent import is_model_consistent
from model.loss.inverse_loss import InverseTransform2D

import torch.nn.functional as F
import cv2 





class SETrainer:

    def __init__(self):
        self.args = parse_args()    
        self.project_dir =  osp.join(osp.dirname(osp.abspath(__file__)),"networks",time.strftime("%Y-%m-%d-%H:%M:%s",time.gmtime(time.time())))
        self.save_dir = osp.join(self.project_dir,'checkpoints')
    
        cudnn.benchmark = self.args.cudnn_benchmark

        if not osp.exists(self.save_dir):
            os.makedirs(self.save_dir)

        os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpu_ids        
        
            
        if self.args.wandb:
            self.init_wandb()

        
        self.log(f'save path : {self.save_dir}')
        self.log(f"bg_weight = {self.args.bg_weight},rind_weight = {self.args.rind_weight} ")
   
        # port = int(1e4+np.random.randint(1,10000,1)[0])
        # logger.info(f"port == {port}")
        # os.environ['MASTER_ADDR'] = 'localhost'
        # os.environ['MASTER_PORT'] = str(port)
        # logger.info(f"node number == {args.nprocs}")
        
        
        
    def wandb_log(self,message):
        if self.args.wandb and hasattr(self,'wandb_init') and self.local_rank==0:
            wandb.log(message)

            
    def log(self,message):
        if self.args.local_rank == 0:
            logger.info(message)
        
    '''
    description: 
    criterion2 : 
    return {*}
    '''
    def train_epoch(self, epoch,edge_branch_out="edge"): # transfer_model=None, transfer_optim=None):
        
        task_list_array = [['background'],['depth'],
                        ['normal'],['reflectance'],
                        ['illumination']]
        root_task_list_array = ['background','depth',  'normal',"reflectance",'illumination']

        batch_time_list = list()
        data_time_list = list()
        losses_list = list()
        losses_array_list = list()
        scores_list = list()

        for i in range(5):
            batch_time_list.append(AverageMeter())
            data_time_list.append(AverageMeter())
            losses_list.append(AverageMeter())
            losses_array = list()
            for it in task_list_array[i]:
                losses_array.append(AverageMeter())
            losses_array_list.append(losses_array)
            scores_list.append(AverageMeter())

        self.model.train()

        for i, in_tar_name_pair in enumerate(self.train_loader):#* 一个一个batch取数据

            input = torch.autograd.Variable(in_tar_name_pair[0].cuda(self.args.local_rank) )
            target= torch.autograd.Variable( in_tar_name_pair[1].cuda(self.args.local_rank))

            output = self.model(input)
            
            '''
            description:  edge detection branch output 共两类 , 
            1.  一类 01map也就是 0,1  , header 最后接的是sigmoid   , 输出就是每个像素点是边缘的概率
            2. 一类就是unet ,header 最后是ReLU, 并且需要改成5类,  求最后结果的时候对对第一维求最大值, 就得到model 对5个类别的分类结果
            return {*}
            '''
            loss = None
            b_loss=None
            if edge_branch_out == "edge":
                b_loss = self.edge_atten_criterion([output[0]],target[:,0,:,:].unsqueeze(1))#* (B,N,W,H),(B,N,W,H)
            elif edge_branch_out == "unet" :
                b_loss = self.focal_criterion(output[0],target[:,0,:,:])#* (B,N,W,H),(B,N,W,H)
            else :
                raise Exception('edge_branch_out is invalid:{}'.format(edge_branch_out))
            
            rind_loss = self.rind_atten_criterion(output[1:],target[:,1:,:,:])#* 可以对多个类别计算loss ,但是这里只有一个类别

            if torch.isnan(b_loss) or torch.isnan(rind_loss)  :
                self.log("nan")
                self.log('b_loss is: {0}'.format(b_loss))
                self.log('rind_loss is: {0}'.format(rind_loss)) 
                exit(0)

            
            if self.inverse_form_criterion is not None: 
                #?  background_out 是否需要clone? 
                #*  after detach , performance is better.
                background_out= output[0].clone().detach()
                rind_out = output[1:]
                rind_out_stack_max_value = torch.stack(rind_out).max(0)[0]
                extra_loss = self.inverse_form_criterion(rind_out_stack_max_value,background_out)
                # rind_out_stack_mean_value = torch.stack(rind_out).mean(0)
                # extra_loss = inverse_form_criterion(rind_out_stack_mean_value,background_out)
                loss =  self.args.bg_weight*b_loss+ self.args.rind_weight*rind_loss   + self.args.extra_loss_weight * extra_loss
            else :
                loss =  self.args.bg_weight*b_loss+ self.args.rind_weight*rind_loss  



            self.optimizer.zero_grad()
            loss.backward()#* warning exists
            self.optimizer.step()

            #* print status 
            if  i % self.args.print_freq == 0 and self.args.local_rank == 0:#* for debug 

                if self.inverse_form_criterion is not None: 
                    all_need_upload = { "b_loss":b_loss,"rind_loss":rind_loss,"total_loss":loss,"extra_loss":extra_loss}
                else :
                    all_need_upload = { "b_loss":b_loss,"rind_loss":rind_loss,"total_loss":loss}
                
                self.wandb_log(all_need_upload)

                tmp = 'Epoch: [{0}][{1}/{2}/{3}]'.format(epoch, i, len(self.train_loader),self.args.local_rank)
                tmp+= "\t".join([f"{k} : {v} \t" for k,v in all_need_upload.items()])
                self.log(tmp)

            



    def init_distributed(self):
        
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(self.args.local_rank) 
        torch.autograd.set_detect_anomaly(True) 
        

        
    def init_wandb(self):
        wandb.init(project="train_cerberus") 
        for k, v in self.args.__dict__.items():
            setattr(wandb.config,k,v)
        setattr(wandb.config,"extra_loss_weight",self.args.extra_loss_weight)
        self.wandb_init =True






    def init_model(self):
        #* construct model 
        # single_model = CerberusSegmentationModelMultiHead(backbone="vitb_rn50_384")
        single_model = EdgeCerberus(backbone="vitb_rn50_384")
        #*========================================================
        #* calc parameter numbers 
        total_params,Trainable_params,NonTrainable_params =calculate_param_num(single_model)
        self.log(f"total_params={total_params},Trainable_params={Trainable_params},NonTrainable_params:{NonTrainable_params}")
        #*========================================================

        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(single_model.cuda(self.args.local_rank))
        model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[self.args.local_rank],
                            find_unused_parameters=True,broadcast_buffers = True) 
        

        if self.args.resume:
            if os.path.isfile(self.args.resume):
                checkpoint = torch.load(self.args.resume,map_location=torch.device(self.args.local_rank))
                self.start_epoch = checkpoint['epoch']
                best_prec1 = checkpoint['best_prec1']
                
                for name, param in checkpoint['state_dict'].items():
                    # name = name[7:]
                    model.state_dict()[name].copy_(param)
                self.log("=> loaded checkpoint '{}' (epoch {})".format(self.args.resume, checkpoint['epoch']))
            else:
                self.log("=> no checkpoint found at '{}'".format(self.args.resume))


        elif self.args.pretrained_model:
            if os.path.isfile(self.args.pretrained_model):

                checkpoint = torch.load(self.args.pretrained_model,map_location='cpu')
                for name, param in checkpoint['state_dict'].items():
                    if name[:5] == 'sigma':
                        pass
                        # model.state_dict()[name].copy_(param)
                    else:
                        model.state_dict()[name].copy_(param)

                self.log("=> loaded model checkpoint '{}' (epoch {})".format(self.args.pretrained_model, checkpoint['epoch']))
            else:
                self.log("=> no checkpoint found at '{}'".format(self.args.resume))
        
        

        self.model = model
        

    def inti_criterion(self):
        self.edge_atten_criterion = AttentionLoss2(gamma=self.args.edge_loss_gamma,beta=self.args.edge_loss_beta).cuda(self.args.local_rank)
        self.rind_atten_criterion = AttentionLoss2(gamma=self.args.rind_loss_gamma,beta=self.args.rind_loss_beta).cuda(self.args.local_rank)
        self.focal_criterion = SegmentationLosses(weight=None, cuda=True).build_loss(mode='focal')


        if self.args.constraint_loss:
            self.inverse_form_criterion = InverseTransform2D()
        else :
            self.inverse_form_criterion =None

    def inti_dataloader(self):
            
        train_dataset = Mydataset(root_path=self.args.train_dir, split='trainval', crop_size=self.args.crop_size)

        #!========================
        # sample = train_dataset.__getitem__(0)
        # cv2.imwrite('im.jpg',sample[0].permute(1,2,0).numpy()*255)
        # cv2.imwrite('label1.jpg',sample[1].numpy()[0,:,:]*255)
        #!========================
        self.train_sampler = DistributedSampler(train_dataset)

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,batch_size=self.args.batch_size, num_workers=self.args.workers,
                pin_memory=True, drop_last=True, sampler=self.train_sampler) 

        #* load test data =====================
        test_dataset = Mydataset(root_path=self.args.test_dir, split='test', crop_size=self.args.crop_size)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, 
                                shuffle=False,num_workers=self.args.workers,pin_memory=False)
        
        

    '''
    description:  
    param {*} local_rank : 分布式训练参数 , 考试当前第几个节点,第几个GPU
    param {*} nprocs:  分布式训练参数 , 表示共有几个节点用于计算每个节点的batch size
    param {*} args : 训练脚本的超参数
    return {*}
    '''
    def train(self):
        self.init_distributed()
        
        self.init_model() #* todo 
        self.inti_criterion()
        self.inti_dataloader()
        self.optimizer = torch.optim.SGD(self.model.parameters(),self.args.lr,
                                    momentum=self.args.momentum,weight_decay=self.args.weight_decay)
        
        if not hasattr(self,'start_epoch'):#* perhaps be init during  resuming
            self.start_epoch = 0


        for epoch in range(self.start_epoch, self.args.epochs):
            lr = self.adjust_learning_rate(epoch)

            self.log('Epoch: [{0}]\tlr {1:.06f}'.format(epoch, lr))

            self.train_sampler.set_epoch(epoch)        

            self.train_epoch(epoch)

            self.save_ckpt(epoch)

        self.log("train finish!!!! ")

        

    def save_ckpt(self,epoch):
        #* 要么就每30epoch 保存一次 
        if((epoch % self.args.save_freq == 0 or  epoch+1 == self.args.epochs ) 
            and self.args.local_rank == 0):
            
            checkpoint_path = osp.join(self.save_dir,'ckpt_rank%03d_ep%04d.pth.tar'%(self.args.local_rank,epoch))
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': self.args.arch,
                'state_dict': self.model.state_dict(),
                'best_prec1': None,
            }, True, filename=checkpoint_path)
        


        

    def adjust_learning_rate(self, epoch):

        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

        if self.args.lr_mode == 'step':
            lr = self.args.lr * (0.1 ** (epoch // self.args.step))
        elif self.args.lr_mode == 'poly':
            lr = self.args.lr * (1 - epoch / self.args.epochs) ** 0.9
        else:
            raise ValueError('Unknown lr mode {}'.format(self.args.lr_mode))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        #adjust the learning rate of sigma
        self.optimizer.param_groups[-1]['lr'] = lr * 0.01

        return lr
        

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    
if __name__ == '__main__':

    setup_seed(20)
    trainer = SETrainer()
    trainer.train()


    

    