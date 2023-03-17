

import os
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.autograd import Variable
import torchvision.transforms as transforms
from tqdm import tqdm
from skimage import img_as_ubyte
import cv2




#* utils 
import os.path as osp
from os.path import join,split, isdir,isfile, exists
import random
import wandb
from loguru import logger
import warnings


from utils import save_checkpoint,AverageMeter,calculate_param_num
from utils.semantic_edge_option import parse_args
from utils.utils import *
from utils.global_var import *

from utils.lr_scheduler import get_scheduler



#* loss  function 
from model.loss.inverse_loss import InverseTransform2D
from utils.loss import SegmentationLosses
from utils.edge_loss2 import AttentionLossSEG,AttentionLossSE
from utils.DFF_losses import EdgeDetectionReweightedLossesSingle,EdgeDetectionReweightedLosses



#* dataloader 
from dataloaders.datasets.bsds_hd5 import Mydataset
from dataloaders.semantic_edge import get_edge_dataset


#* model 
from model.ECT_SE import EdgeCerberusMultiClass



# from torchsummary import summary

from  glob import glob

from IPython import embed 
warnings.filterwarnings('ignore')




def namestr(obj, namespace):
    for name in namespace:
        if namespace[name] is obj:
            return name 
    return None


def detach_module(model):
    if len(list(model.keys())[0].split('.')) <=6:
        return model

    new_state_dict = OrderedDict()
    for k, v in model.items():
        name = k[7:] # module字段在最前面，从第7个字符开始就可以去掉module
        new_state_dict[name] = v #新字典的key值对应的value一一对应

    return new_state_dict 






class SETrainer:

    def __init__(self):
        self.args = parse_args()
        if self.args.local_rank==0:
            self.project_dir =  osp.join(osp.dirname(osp.abspath(__file__)),"networks",self.args.dataset,time.strftime("%Y-%m-%d-%H:%M:%s",time.gmtime(time.time())))
            
            self.save_dir = osp.join(self.project_dir,'checkpoints')
            if not osp.exists(self.save_dir):
                os.makedirs(self.save_dir)
            self.log(f'save path : {self.save_dir}')
        
        cudnn.benchmark = self.args.cudnn_benchmark
        # os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpu_ids    
        
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(self.args.local_rank) 
        torch.autograd.set_detect_anomaly(True)     
   
        # port = int(1e4+np.random.randint(1,10000,1)[0])
        # logger.info(f"port == {port}")
        # os.environ['MASTER_ADDR'] = 'localhost'
        # os.environ['MASTER_PORT'] = str(port)
        # logger.info(f"node number == {args.nprocs}")

        #* record the performance on val subset 
        self.best_se_edge_loss  = 1e+6
        self.current_se_edge_loss = 1e+6

        
        self.inti_dataloader()
        self.init_model() #* todo 
        
        if self.args.cmd=='test':
            return 

        self.inti_criterion()
        if not hasattr(self,'start_epoch'):#* perhaps be init during  resuming
            self.start_epoch = 0

        if self.args.wandb and self.args.local_rank ==0:
            self.init_wandb()


    def get_mode(self):

        return self.args.cmd
        

    def log_file(self,message):
        
        if self.args.local_rank == 0 :
            with open(join(self.project_dir ,'log.txt'),'a') as f :
                f.write(message)
            
        
        
        
    def wandb_log(self,message,step=None):
        if self.args.wandb and hasattr(self,'wandb_init') and self.args.local_rank==0:
            if step is not None:
                # wandb.log(message,step = step)
                message.update({'step':step})
                wandb.log(message)
            else:
                wandb.log(message)

            
    def log(self,message):
        if self.args.local_rank == 0:
            logger.info(message)
        
  
    def init_wandb(self):
        wandb.init(project="semantic_edge_cerberus") 
        
        message=""
        for k, v in self.args.__dict__.items():
            setattr(wandb.config,k,v)
            message+= f"{k}: {v} \n"
            
        self.log_file(message)
        setattr(wandb.config,'save_directory',self.save_dir)

        self.wandb_init =True


    '''
    description:  inti model with schedule and optimizer 
    param {*} self
    return {*}
    '''
    def init_model(self):
        #* construct model 
        # single_model = CerberusSegmentationModelMultiHead(backbone="vitb_rn50_384")
        
        
        if self.args.dataset == 'bsds':
            single_model = EdgeCerberusMultiClass(backbone="vitb_rn50_384")
            self.class_num = 4
        elif self.args.dataset == 'cityscapes' :
            self.class_num = 19
            # single_model = SEdgeCerberus(backbone="vitb_rn50_384",hard_edge_cls_num=self.class_num)
            single_model = EdgeCerberusMultiClass(backbone="vitb_rn50_384",hard_edge_cls_num=self.class_num)
        elif self.args.dataset == 'sbd':
            self.class_num = 20
            #*  for val 
            # single_model = SEdgeCerberus(backbone="vitb_rn50_384",hard_edge_cls_num=self.class_num)
            single_model = EdgeCerberusMultiClass(backbone="vitb_rn50_384",hard_edge_cls_num=self.class_num)
            self.log(f" edge cerberus v2 is loaded ")
            
            

        params_list = [{'params': single_model.pretrained.parameters(), 'lr': self.args.lr},
                        {'params': single_model.scratch.parameters(), 'lr': self.args.lr * 10},
                        {'params': single_model.edge_query_embed.parameters(), 'lr': self.args.lr * 10},
                        {'params': single_model.decoder.parameters(), 'lr': self.args.lr * 10},
                        {'params': single_model.final_norm1.parameters(), 'lr': self.args.lr * 10},
                        {'params': single_model.final_dropout1.parameters(), 'lr': self.args.lr * 10},
                        {'params': single_model.final_rcu.parameters(), 'lr': self.args.lr * 10}]


                           
        # self.optimizer = torch.optim.SGD(self.model.parameters(),self.args.lr,
        #                             momentum=self.args.momentum,\
        #                                 weight_decay=self.args.weight_decay)

        if self.args.cmd != 'test':
            self.optimizer = torch.optim.SGD(params_list,self.args.lr,
                                        momentum=self.args.momentum,\
                                        weight_decay=self.args.weight_decay)
            
            self.scheduler = get_scheduler(self.optimizer, len(self.train_loader), self.args)

 
        #*========================================================
        #* calc parameter numbers 
        # self.log(summary(single_model))
        #*========================================================

        
        #* distributed train 
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(single_model.cuda(self.args.local_rank))
        model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[self.args.local_rank],
                            find_unused_parameters=True,broadcast_buffers = True) 
        
        self.model = model
        
        if self.args.resume:
            if os.path.isfile(self.args.resume):
                self.load_checkpoint(self.args.resume)
            else:
                self.log("=> no checkpoint found at '{}'".format(self.args.resume))
        elif self.args.pretrained_model:
            if os.path.isfile(self.args.pretrained_model):
                checkpoint = torch.load(self.args.pretrained_model,map_location='cpu')
                for name, param in checkpoint['state_dict'].items():
                    if name[:5] == 'sigma':
                        pass
                    else:
                        self.model.state_dict()[name].copy_(param)
                self.log("=> loaded model checkpoint '{}' (epoch {})".format(self.args.pretrained_model, checkpoint['epoch']))
            else:
                self.log("=> no checkpoint found at '{}'".format(self.args.resume))
            del checkpoint
            torch.cuda.empty_cache()
            
            
            
        #* update lr decay milestones
        #* load model 之后 schedule 也变了 , 变成上次训练的,这次的就不见了, 重新加载
        if self.args.change_decay_epoch :#* update lr decay epoch 
            #* last epoch == last counter (bug)...
            self.log(f"scheduler.milestones : {self.scheduler.milestones} \t current step :{self.scheduler._step_count} \t last epoch {self.scheduler.last_epoch} ")
            self.log(f"args.lr_decay_epochs :{self.args.lr_decay_epochs} \t len(train_loader):{len(self.train_loader)}")
            
            self.scheduler.milestones ={len(self.train_loader)*( l+1 - self.start_epoch )+self.scheduler.last_epoch : 1 for l in self.args.lr_decay_epochs}
            
            self.log(f"scheduler.milestones : {self .scheduler.milestones} \t current step :{self.scheduler._step_count} \t last epoch {self.scheduler.last_epoch} ")
            
            
            self.log(f"scheduler.gamma : {self.scheduler.gamma} ")
            self.scheduler.gamma = self.args.lr_decay_rate
            self.log(f"scheduler.gamma : {self.scheduler.gamma} ")
            
                




    def load_checkpoint(self, ckpt_path):
        """Load from checkpoint."""
        
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        
        self.start_epoch = int(checkpoint['epoch']) + 1 #* train next epoch 
        
        #todo checkpoint['model']  delete ".module"
        # if distributed2common:
            # common_model = detach_module(checkpoint['model'])
        # list(common_model.keys())[0]
            # self.model.load_state_dict(common_model, True)
        # else :
        #     model.load_state_dict(checkpoint['model'], True)

        self.model.load_state_dict(checkpoint['model'], True)
        
        if self.args.cmd != 'test':
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.log(f" loaded schedule lr : {self.scheduler.get_last_lr()[0]}")
        
        
        self.best_se_edge_loss  =checkpoint['val_se_edge_loss']
        self.current_se_edge_loss = checkpoint['val_se_edge_loss']

        self.log("=> loaded successfully '{}' (epoch {})".format(
            ckpt_path, checkpoint['epoch']
        ))
        
        
        del checkpoint
        torch.cuda.empty_cache()


    def inti_criterion(self):
        # self.edge_criterion = AttentionLossSEG(gamma=self.args.edge_loss_gamma,beta=self.args.edge_loss_beta).cuda(self.args.local_rank)
        self.edge_criterion = EdgeDetectionReweightedLosses().cuda(self.args.local_rank)
        

        if self.args.dataset == 'cityscapes':
            self.hard_edge_criterion = EdgeDetectionReweightedLosses().cuda(self.args.local_rank)
        elif self.args.dataset == 'bsds':
            self.hard_edge_criterion = AttentionLoss2(gamma=self.args.rind_loss_gamma,beta=self.args.rind_loss_beta).cuda(self.args.local_rank)
        elif self.args.dataset == 'sbd':
            self.hard_edge_criterion = EdgeDetectionReweightedLosses().cuda(self.args.local_rank)
            # self.hard_edge_criterion = AttentionLossSE(gamma=self.args.rind_loss_gamma,beta=self.args.rind_loss_beta).cuda(self.args.local_rank)



        # self.focal_criterion = SegmentationLosses(weight=None, cuda=True).build_loss(mode='focal')
        
        self.inverse_form_criterion = InverseTransform2D()
        

    def inti_dataloader(self):

        def wrapper_loader(dataset,batch_size,workers):
            sampler = DistributedSampler(dataset)
            loader = torch.utils.data.DataLoader(
                dataset,batch_size=batch_size, num_workers=workers,
                    pin_memory=True, drop_last=True, sampler=sampler) 
            return loader,sampler
        

        if self.args.dataset == 'bsds':
            train_dataset = Mydataset(root_path=self.args.train_dir, \
                split='trainval', crop_size=self.args.crop_size)
            test_dataset = Mydataset(root_path=self.args.test_dir, split='test', crop_size=self.args.crop_size)

        elif self.args.dataset == 'cityscapes' or self.args.dataset == 'sbd':
            input_transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(
                                [.485, .456, .406],
                                [.229, .224, .225]
                            )
                        ]
                    )

            #* the meaning of scale : 'choose to use random scale transform(0.75-2),default:multi scale')
            #* if test the SBD, the crop_size should be set as  512, otherwise  origin size for cityscapes
            test_dataset = get_edge_dataset(self.args.dataset,  split='val', mode='testval',
                                    transform=input_transform, crop_size=self.args.crop_size,root=self.args.data_dir)

            self.test_loader = torch.utils.data.DataLoader(test_dataset, 
                                    batch_size=self.args.batch_size, shuffle=False,
                                    num_workers=self.args.workers)
            
            if self.args.cmd == 'test':
                return 

            #* not base size need
            data_kwargs = {'transform': input_transform, 'base_size': self.args.crop_size,
                'crop_size': self.args.crop_size, 'logger': logger,
                'scale': True,"root":self.args.data_dir}

            train_dataset  = get_edge_dataset(self.args.dataset, split='train', mode='train',
                                                **data_kwargs)

            val_dataset  = get_edge_dataset(self.args.dataset, split='val', mode='val',**data_kwargs)



        
        self.train_loader,self.train_sampler = wrapper_loader(train_dataset,
                                    self.args.batch_size,self.args.workers)


        self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.args.batch_size, 
                                                      shuffle=False, num_workers=self.args.workers)
            
        #!========================
        # sample = train_dataset.__getitem__(0)
        # cv2.imwrite('im.jpg',sample[0].permute(1,2,0).numpy()*255)
        # cv2.imwrite('label1.jpg',sample[1].numpy()[0,:,:]*255)
        #!========================
        

    '''
    description:  
    param {*} local_rank : 分布式训练参数 , 考试当前第几个节点,第几个GPU
    param {*} nprocs:  分布式训练参数 , 表示共有几个节点用于计算每个节点的batch size
    param {*} args : 训练脚本的超参数
    return {*}
    '''
    def train(self):

        for epoch in range(self.start_epoch, self.args.epochs):

            self.train_sampler.set_epoch(epoch)
            self.train_epoch(epoch)

            self.scheduler.step(epoch)
            self.log(f"after update, lr == {self.scheduler.get_last_lr()[0]}")
        
            
            if epoch >= self.args.val_all_in:
                self.args.val_freq  = 1
                self.args.save_freq = 1   
                self.log('update val_freq and save freq')
                

            if epoch % self.args.val_freq == 0 and self.args.local_rank == 0 :
                self.validate_epoch(epoch)

                if (epoch % self.args.save_freq == 0 or  epoch+1 == self.args.epochs ):
                    self.save_ckpt(epoch)

        self.log("train finish!!!! ")

        

    def save_ckpt(self,epoch):
        checkpoint_path = osp.join(self.save_dir,'ckpt_rank%03d_ep%04d.pth.tar'%(self.args.local_rank,epoch))
        
        save_checkpoint({
            'args': self.args,
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'val_se_edge_loss': self.current_se_edge_loss,
        }, self.is_best, 
        filename=checkpoint_path)


        
        


    '''
    description:   the  funciton of  this block has been replaced by schedule
    param {*} self
    param {*} epoch
    return {*}
    '''
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
        # self.optimizer.param_groups[-1]['lr']
        return lr

    def validate_epoch(self,epoch):
        self.model.eval()
        se_loss = 0
        
        
        
        for i, (input, target) in enumerate(self.val_loader):#* 一个一个batch取数据

            input = input.cuda()
            target = target.cuda()

            with torch.no_grad():
                # output = self.model(input)
                output = self.model(input.type(torch.FloatTensor))
            
                #* for cityscapes
               
                if self.args.dataset == 'bsds':
                    b_loss = self.edge_criterion([output[0]],target[:,0,:,:].unsqueeze(1))#* (B,N,W,H),(B,N,W,H)
                elif self.args.dataset == 'cityscapes' or self.args.dataset == 'sbd':
                    generic_edge,indices = torch.max(target[:,1:,:,:],1)
                    pred_edge = F.sigmoid(output[0])
                    # cv2.imwrite('a.jpg',generic_edge.squeeze().cpu().numpy()*255)
                    b_loss = self.edge_criterion([pred_edge],generic_edge.unsqueeze(1))#* (B,N,W,H),(B,N,W,H)
        
             
                if self.args.dataset == 'bsds':
                    rind_loss = self.hard_edge_criterion(output[1:],target[:,1:,:,:])#* 可以对多个类别计算loss ,但是这里只有一个类别
                elif self.args.dataset == 'cityscapes' :
                    # rind_loss = self.hard_edge_criterion(torch.cat(output[1:],dim=1),target[:,1:,:,:])#* 可以对多个类别计算loss ,但是这里只有一个类别
                    hard_prediction_maps = torch.cat(output[1:],dim=1)
                    rind_loss = self.hard_edge_criterion(hard_prediction_maps ,target)#* 可以对多个类别计算loss ,但是这里只有一个类别
                elif self.args.dataset == 'sbd' :
                    rind_loss = self.hard_edge_criterion(output[1:] ,target)#* 可以对多个类别计算loss ,但是这里只有一个类别
                    

                
                if self.inverse_form_criterion is not None: 
                    #?  background_out 是否需要clone? 
                    #*  after detach , performance is better.
                    #* why detach?  using as 
                    

                    if self.args.dataset == 'bsds':
                        background_out= output[0].clone().detach()
                        rind_out = output[1:]
                        rind_out_stack_max_value = torch.stack(rind_out).max(0)[0]
                        inverse_form_loss = self.inverse_form_criterion(rind_out_stack_max_value,background_out)
                    elif self.args.dataset == 'cityscapes' :
                        background_out = pred_edge.clone().detach()
                        hard_edge_merged,__ = torch.max(hard_prediction_maps,1)
                        inverse_form_loss1 = self.inverse_form_criterion(hard_edge_merged.unsqueeze(1),background_out)
                        inverse_form_loss =  inverse_form_loss1
                    elif self.args.dataset == 'sbd' :
                        background_out = pred_edge.clone().detach()

                        hard_edge_merged,__ = torch.max( torch.cat(output[1:],dim=1),1)
                        inverse_form_loss1 = self.inverse_form_criterion(hard_edge_merged.unsqueeze(1),background_out)
                        inverse_form_loss =  inverse_form_loss1
                        

                    loss =  b_loss + rind_loss +inverse_form_loss
                else :
                    loss =  b_loss+ rind_loss  

                se_loss+=rind_loss.item()

                all_need_upload = { "val_generic_edge_loss":b_loss.item(),"val_hard_edge_loss":rind_loss.item(),"val_total_loss":loss.item(),'val_step':epoch*len(self.val_loader)+i}
                # all_need_upload = { "val_generic_edge_loss":b_loss.item(),"val_hard_edge_loss":rind_loss.item(),"val_total_loss":loss.item()}


                if 'inverse_form_loss1' in locals():
                    all_need_upload['val_inverse_form_loss1'] = inverse_form_loss1.item()

                    
                if (i+1) % self.args.print_freq  == 0 :

                    # self.wandb_log(all_need_upload,step=epoch*len(self.val_loader)+i)
                    self.wandb_log(all_need_upload)

                    tmp = 'Validation Epoch: [{0}][{1}/{2}/{3}]'.format(epoch, i, len(self.train_loader),self.args.local_rank) + \
                                "\t".join([f"{k} : {v} \t" for k,v in all_need_upload.items()])
                    self.log(tmp)

        

        self.log_file("epoch %d \t val loss: %f \t ")
        self.current_se_edge_loss = se_loss
        if self.current_se_edge_loss < self.best_se_edge_loss:
            self.best_se_edge_loss = self.current_se_edge_loss
            self.is_best = True
            self.wandb_log({'best_model_epoch':epoch})

            self.log_file("epoch %d \t val loss: %f \t is best")

        else:
            self.is_best = False
            
            

    
    def validate_all_model(self):

        if self.args.resume_model_dir == None:
            self.log('please give the ckpt directory ')
            return 
        
        # example_dir = "/home/DISCOVER_summer2022/xusc/exp/cerberus/networks/2023-02-26-13:27:1677389259/checkpoints/ckpt_*"
        
        all_models = sorted(glob(join(self.args.resume_model_dir ,"ckpt_*")))
        # all_models =[ join(self.args.resume_model_dir ,"model_best.pth.tar")]

        
        self.log(all_models)
        self.args.print_freq = 1e+10

        for model_file in tqdm(all_models):

            self.load_checkpoint(model_file)
            self.log(f'current loss : {self.current_se_edge_loss }')
            
            
            # tic = time.time()
            # self.validate_epoch(self.start_epoch)
            # spend_time = time.strftime("%H:%M:%S",time.gmtime(time.time()-tic))
            # self.log(f" ckpt :  {model_file.split('/')[-1]} \t val loss : {self.current_se_edge_loss} \t spend time : {spend_time}")
            
            if hasattr(self,'is_best') and self.is_best :
                self.log('current model  is better')
            self.log("==============================================================================================================================================")


    def train_epoch(self, epoch,edge_branch_out="edge"): # transfer_model=None, transfer_optim=None):
        
        # task_list_array = [['background'],['depth'],
        #                 ['normal'],['reflectance'],
        #                 ['illumination']]
        # root_task_list_array = ['background','depth',  'normal',"reflectance",'illumination']

        # batch_time_list = list()
        # data_time_list = list()
        # losses_list = list()
        # losses_array_list = list()
        # scores_list = list()

        # for i in range(5):
        #     batch_time_list.append(AverageMeter())
        #     data_time_list.append(AverageMeter())
        #     losses_list.append(AverageMeter())
        #     losses_array = list()
        #     for it in task_list_array[i]:
        #         losses_array.append(AverageMeter())
        #     losses_array_list.append(losses_array)
        #     scores_list.append(AverageMeter())

        self.model.train()
        

        for i, in_tar_name_pair in enumerate(self.train_loader):#* 一个一个batch取数据

            input = torch.autograd.Variable(in_tar_name_pair[0].cuda(self.args.local_rank) )
            target= torch.autograd.Variable( in_tar_name_pair[1].cuda(self.args.local_rank))
            
            # output = self.model(input)
            output = self.model(input.type(torch.FloatTensor))
            
            '''
            description:  edge detection branch output 共两类 , 
            1.  一类 01map也就是 0,1  , header 最后接的是sigmoid   , 输出就是每个像素点是边缘的概率
            2. 一类就是unet ,header 最后是ReLU, 并且需要改成5类,  求最后结果的时候对对第一维求最大值, 就得到model 对5个类别的分类结果
            return {*}
            '''
            if edge_branch_out == "edge":
                #* for cityscapes
                
            
                if self.args.dataset == 'bsds' :
                    b_loss = self.edge_criterion([output[0]],target[:,0,:,:].unsqueeze(1))#* (B,N,W,H),(B,N,W,H)

                elif self.args.dataset == 'cityscapes' or self.args.dataset == 'sbd' :
                    generic_edge,indices = torch.max(target[:,1:,:,:],1)

                    
                    
                    # pred_edge = F.sigmoid(output[0])
                    # b_loss = self.edge_criterion([pred_edge],generic_edge.unsqueeze(1))#* (B,N,W,H),(B,N,W,H)
                    
                    b_loss = self.edge_criterion(
                                        output[0],
                                        torch.cat([target[:,0:1,:,:],generic_edge.unsqueeze(1)],axis=1)
                                    ) #* (B,N,W,H),(B,N,W,H)
                
                    

            elif edge_branch_out == "unet" :
                b_loss = self.focal_criterion(output[0],target[:,0,:,:])#* (B,N,W,H),(B,N,W,H)
            else :
                raise Exception('edge_branch_out is invalid:{}'.format(edge_branch_out))
         
            if self.args.dataset == 'bsds':
                rind_loss = self.hard_edge_criterion(output[1:],target[:,1:,:,:])#* 可以对多个类别计算loss ,但是这里只有一个类别
            elif self.args.dataset == 'cityscapes' :
                # rind_loss = self.hard_edge_criterion(torch.cat(output[1:],dim=1),target[:,1:,:,:])#* 可以对多个类别计算loss ,但是这里只有一个类别
                hard_prediction_maps = torch.cat(output[1:],dim=1)
                rind_loss = self.hard_edge_criterion(hard_prediction_maps ,target)
            
            elif  self.args.dataset == 'sbd':
                rind_loss = self.hard_edge_criterion(output[1:] ,target)#* 可以对多个类别计算loss ,但是这里只有一个类别

                

            if torch.isnan(b_loss) or torch.isnan(rind_loss)  :
                self.log("nan")
                self.log('b_loss is: {0}'.format(b_loss))
                self.log('rind_loss is: {0}'.format(rind_loss)) 
                exit(0)

            
        
            #?  background_out 是否需要clone? 
            #*  after detach , performance is better.
            #* why detach?  using as 
            

            if self.args.dataset == 'bsds':
                background_out= output[0].clone().detach()
                rind_out = output[1:]
                rind_out_stack_max_value = torch.stack(rind_out).max(0)[0]
                extra_loss = self.inverse_form_criterion(rind_out_stack_max_value,background_out)

            elif self.args.dataset == 'cityscapes' :

                background_out = pred_edge.clone().detach()
                hard_edge_merged,__ = torch.max(hard_prediction_maps,1)
                inverse_form_loss1 = self.inverse_form_criterion(hard_edge_merged.unsqueeze(1),background_out)
                inverse_form_loss =  inverse_form_loss1

            elif  self.args.dataset == 'sbd':
                background_out = F.sigmoid(output[0]).clone().detach()
                hard_edge_merged,__ = torch.max( torch.cat(output[1:],dim=1),1)
                inverse_form_loss1 = self.inverse_form_criterion(hard_edge_merged.unsqueeze(1),background_out)
                inverse_form_loss =  inverse_form_loss1


            loss =  self.args.bg_weight*b_loss+ self.args.rind_weight*rind_loss \
                        + self.args.inverseform_loss_weight * inverse_form_loss
        
            
            


            self.optimizer.zero_grad()
            loss.backward()#* warning exists
            self.optimizer.step()
            # self.scheduler.step()
            # self.scheduler.step(epoch)

            #* print status 
            if  i % self.args.print_freq == 0 and self.args.local_rank == 0:#* for debug 
                # embed()
                
                # self.log(f'scheduler step counter : {self.scheduler._step_count} last epoch: {self.scheduler.last_epoch} milestones: {self.scheduler.milestones} current lr: {self.scheduler.get_last_lr()[0]};')
                
                #* get_last_lr return the learning rate of each parameter group
                all_need_upload = { "generic_edge_loss":b_loss.item(),"hard_edge_loss":rind_loss.item(),"total_loss":loss.item(),
                                    'lr':self.scheduler.get_last_lr()[0],'train_step':epoch*len(self.train_loader)+i,'epoch':epoch}
                

                # locals() : 基于字典的访问局部变量的方式。键是变量名，值是变量值。
                # globals() : 基于字典的访问全局变量的方式。键是变量名，值是变量值。

                if 'inverse_form_loss1' in locals():
                    all_need_upload['inverse_form_loss1'] = inverse_form_loss1.item()

                if 'inverse_form_loss2' in locals():
                    all_need_upload['inverse_form_loss2'] = inverse_form_loss2.item()
                    
                
                
                # self.wandb_log(all_need_upload,step=)
                self.wandb_log(all_need_upload)

                tmp = 'Epoch: [{0}][{1}/{2}/{3}]'.format(epoch, i, len(self.train_loader),self.args.local_rank) + \
                            "\t".join([f"{k} : {v} \t" for k,v in all_need_upload.items()])
                self.log(tmp)

    def test(self):

        self.model.eval()
        
        outdir_list_fuse = []
        #* only one gpu is needed during inference, otherwise the code below would be error 
        for i in range(self.test_loader.dataset.num_class):
            outdir_fuse = '%s/%s/%s/class_%03d'%(self.args.dataset,'cerberus',time.strftime("%H:%M:%s",time.gmtime(time.time())), i+1)
            # if not os.path.exists(outdir_fuse):
            if os.path.exists(outdir_fuse):
                shutil.rmtree(outdir_fuse)
            os.makedirs(outdir_fuse)
            outdir_list_fuse.append(outdir_fuse)
        
        
        tbar = tqdm(self.test_loader, desc='\r')

        for i, (image, im_paths, im_sizes)  in enumerate(tbar):#* 一个一个batch取数据

            # if self.args.dataset != 'cityscapes':
            #     W,H =im_sizes.squeeze()
            #     trans1 = transforms.Compose([transforms.Resize(size=(H//8*8, W//8*8))])
            #     trans2 = transforms.Compose([transforms.Resize(size=(H, W))])
            #     image = trans1(image)
            
            image = image.type(torch.FloatTensor)
            
            with torch.no_grad():    
                fuses = self.model(image)

            fuses = torch.cat(fuses[1:],dim=1)#* filter the background edge 
                
            def save_one_prediction(predictions,out_cls_dir,impath):
                for i in range(predictions.shape[0]):
                    path = os.path.join(out_cls_dir[i], impath)
                    cv2.imwrite(path, img_as_ubyte(predictions[i]))

            # if self.args.dataset != 'cityscapes':
            #     side5s = trans2(side5s)
            #     fuses = trans2( fuses)

                    
            #* traverse each batch size 
            for idx in range(im_sizes.shape[0]): 
                im_size = im_sizes[idx]
                fuse = fuses[idx]
                
                fuse =  fuse.squeeze_().sigmoid_().cpu().numpy()    
                fuse = fuse[:,0:im_size[1],0:im_size[0]]
                
                # to_imger = transforms.ToPILImage()
                # to_imger(normalize(trans2(image[idx]))).save('img.jpg')
                # save_prediction_for_debug(side5,suffix='side5')
                # save_prediction_for_debug(fuse,suffix='fuse')

                impath = im_paths[idx]
                save_one_prediction(fuse,outdir_list_fuse,impath)

          

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    
if __name__ == '__main__':

    setup_seed(20)
    trainer = SETrainer()
    if trainer.get_mode() == 'test':
        trainer.test()
    elif trainer.get_mode() == 'train':
        trainer.train()
    elif trainer.get_mode() == 'val':
        trainer.validate_all_model()
    


    

    