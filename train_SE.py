

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




#* loss  function 
from model.loss.inverse_loss import InverseTransform2D
from utils.loss import SegmentationLosses
from utils.edge_loss2 import AttentionLoss2
from utils.DFF_losses import EdgeDetectionReweightedLossesSingle,EdgeDetectionReweightedLosses



#* dataloader 
from dataloaders.datasets.bsds_hd5 import Mydataset
from dataloaders.semantic_edge import get_edge_dataset


#* model 
from model.edge_model import EdgeCerberus
# from model.semantic_edge_model import SEdgeCerberus
from model.semantic_edge_model2 import SEdgeCerberus

from torchsummary import summary

from  glob import glob



from IPython import embed 
warnings.filterwarnings('ignore')

def namestr(obj, namespace):
    for name in namespace:
        if namespace[name] is obj:
            return name 
    return None




class SETrainer:

    def __init__(self):
        self.args = parse_args()
        if self.args.local_rank==0:
            self.project_dir =  osp.join(osp.dirname(osp.abspath(__file__)),"networks",time.strftime("%Y-%m-%d-%H:%M:%s",time.gmtime(time.time())))
            self.save_dir = osp.join(self.project_dir,'checkpoints')
            if not osp.exists(self.save_dir):
                os.makedirs(self.save_dir)
            self.log(f'save path : {self.save_dir}')
        
        cudnn.benchmark = self.args.cudnn_benchmark
        os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpu_ids        
   
        # port = int(1e4+np.random.randint(1,10000,1)[0])
        # logger.info(f"port == {port}")
        # os.environ['MASTER_ADDR'] = 'localhost'
        # os.environ['MASTER_PORT'] = str(port)
        # logger.info(f"node number == {args.nprocs}")

        #* record the performance on val subset 
        self.last_se_edge_loss = 1e+6
        self.current_se_edge_loss = 1e+6

        self.init_distributed()
        self.init_model() #* todo 
        self.inti_dataloader()
        if self.args.cmd=='test':
            return 

        self.inti_criterion()
        if not hasattr(self,'start_epoch'):#* perhaps be init during  resuming
            self.start_epoch = 0

        if self.args.wandb and self.args.local_rank ==0:
            self.init_wandb()


    def get_mode(self):

        return self.args.cmd
        
        
        
    def wandb_log(self,message):
        if self.args.wandb and hasattr(self,'wandb_init') and self.args.local_rank==0:
            wandb.log(message)

            
    def log(self,message):
        if self.args.local_rank == 0:
            logger.info(message)
        
  



    def init_distributed(self):
        
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(self.args.local_rank) 
        torch.autograd.set_detect_anomaly(True) 
        

        
    def init_wandb(self):
        wandb.init(project="semantic_edge_cerberus") 

        for k, v in self.args.__dict__.items():
            setattr(wandb.config,k,v)
        setattr(wandb.config,'save_directory',self.save_dir)

        self.wandb_init =True


    def init_model(self):
        #* construct model 
        # single_model = CerberusSegmentationModelMultiHead(backbone="vitb_rn50_384")
        
        if self.args.dataset == 'cityscapes':
            single_model = SEdgeCerberus(backbone="vitb_rn50_384")
        elif self.args.dataset == 'bsds' :
            single_model = EdgeCerberus(backbone="vitb_rn50_384")


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

        self.optimizer = torch.optim.SGD(params_list,self.args.lr,
                                    momentum=self.args.momentum,\
                                    weight_decay=self.args.weight_decay)


        #*========================================================
        #* calc parameter numbers 
        # total_params,Trainable_params,NonTrainable_params =calculate_param_num(single_model)
        # self.log(f"total_params={total_params},Trainable_params={Trainable_params},NonTrainable_params:{NonTrainable_params}")
        # self.log(summary(single_model))
        #*========================================================

        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(single_model.cuda(self.args.local_rank))
        model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[self.args.local_rank],
                            find_unused_parameters=True,broadcast_buffers = True) 
        


        if self.args.resume:
            if os.path.isfile(self.args.resume):
                checkpoint = torch.load(self.args.resume,map_location=torch.device('cpu'))
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

    def update_model(self,file):
        if os.path.isfile(file):
            checkpoint = torch.load(file,map_location=torch.device('cpu'))
            self.start_epoch = checkpoint['epoch']
            # best_prec1 = checkpoint['best_prec1']
            for name, param in checkpoint['state_dict'].items():
                self.model.state_dict()[name].copy_(param)
            self.log("model update successful => loaded checkpoint '{}' (epoch {})".format(self.args.resume, checkpoint['epoch']))
        else:
            self.log("=> no checkpoint found at '{}'".format(self.args.resume))

            

    def inti_criterion(self):
        self.edge_criterion = AttentionLoss2(gamma=self.args.edge_loss_gamma,beta=self.args.edge_loss_beta).cuda(self.args.local_rank)

        if self.args.dataset == 'cityscapes':
            self.class_num = 19
            self.hard_edge_criterion = EdgeDetectionReweightedLosses().cuda(self.args.local_rank)
        elif self.args.dataset == 'bsds':
            self.class_num = 1
            self.hard_edge_criterion = AttentionLoss2(gamma=self.args.rind_loss_gamma,beta=self.args.rind_loss_beta).cuda(self.args.local_rank)
        elif self.args.dataset == 'sbd':
            self.class_num = 20
            self.hard_edge_criterion = EdgeDetectionReweightedLosses().cuda(self.args.local_rank)


        # self.focal_criterion = SegmentationLosses(weight=None, cuda=True).build_loss(mode='focal')
        if self.args.inverseform_loss:
            self.inverse_form_criterion = InverseTransform2D()
        else :
            self.inverse_form_criterion =None

    def inti_dataloader(self):

        if self.args.dataset == 'bsds':
            train_dataset = Mydataset(root_path=self.args.train_dir, \
                split='trainval', crop_size=self.args.crop_size)
            test_dataset = Mydataset(root_path=self.args.test_dir, split='test', crop_size=self.args.crop_size)

        elif self.args.dataset == 'cityscapes':
             
            input_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([.485, .456, .406], [.229, .224, .225])])

            #* the meaning of scale : 'choose to use random scale transform(0.75-2),default:multi scale')
            data_kwargs = {'transform': input_transform, 'base_size': self.args.crop_size,
                            'crop_size': self.args.crop_size, 'logger': logger,
                            'scale': True,"root":self.args.data_dir}

            

            test_dataset  = get_edge_dataset(self.args.dataset, split='val', mode='testval',**data_kwargs)

            self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False,
                                      num_workers=self.args.workers)
            

            if self.args.cmd == 'test':
                return 

            train_dataset  = get_edge_dataset(self.args.dataset, split='train', mode='train',
                                                **data_kwargs)

            val_dataset  = get_edge_dataset(self.args.dataset, split='val', mode='val',**data_kwargs)


        def wrapper_loader(dataset,batch_size,workers):
            sampler = DistributedSampler(dataset)
            loader = torch.utils.data.DataLoader(
                dataset,batch_size=batch_size, num_workers=workers,
                    pin_memory=True, drop_last=True, sampler=sampler) 
            return loader,sampler
        
        
        self.train_loader,self.train_sampler = wrapper_loader(train_dataset,
                                    self.args.batch_size,self.args.workers)


        self.val_loader,self.val_sampler = wrapper_loader(val_dataset,
                                    self.args.batch_size,self.args.workers)

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

            lr = self.adjust_learning_rate(epoch)
            self.log('Epoch: [{0}]\tlr {1:.06f}'.format(epoch, lr))

            self.train_sampler.set_epoch(epoch)        
            self.val_sampler.set_epoch(epoch) 

            
            self.train_epoch(epoch)

            if epoch % self.args.val_freq == 0:
                self.validate_epoch(epoch)

                if((epoch % self.args.save_freq == 0 or  epoch+1 == self.args.epochs ) \
                    and self.args.local_rank == 0):
                    self.save_ckpt(epoch)

        self.log("train finish!!!! ")

        

    def save_ckpt(self,epoch):
        checkpoint_path = osp.join(self.save_dir,'ckpt_rank%03d_ep%04d.pth.tar'%(self.args.local_rank,epoch))
        
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'best_prec1': self.current_se_edge_loss,
        }, self.current_se_edge_loss < self.last_se_edge_loss, 
        filename=checkpoint_path)


        
        
    


        

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
                if self.args.dataset == 'cityscapes':
                    generic_edge,indices = torch.max(target[:,1:,:,:],1)
                    pred_edge = F.sigmoid(output[0])
                    # cv2.imwrite('a.jpg',generic_edge.squeeze().cpu().numpy()*255)
                    b_loss = self.edge_criterion([pred_edge],generic_edge.unsqueeze(1))#* (B,N,W,H),(B,N,W,H)
                else:
                    b_loss = self.edge_criterion([output[0]],target[:,0,:,:].unsqueeze(1))#* (B,N,W,H),(B,N,W,H)
        
                if self.args.dataset == 'cityscapes':
                    # rind_loss = self.hard_edge_criterion(torch.cat(output[1:],dim=1),target[:,1:,:,:])#* 可以对多个类别计算loss ,但是这里只有一个类别
                    hard_prediction_maps = torch.cat(output[1:],dim=1)
                    rind_loss = self.hard_edge_criterion(hard_prediction_maps ,target)#* 可以对多个类别计算loss ,但是这里只有一个类别
                elif self.args.dataset == 'bsds':
                    rind_loss = self.hard_edge_criterion(output[1:],target[:,1:,:,:])#* 可以对多个类别计算loss ,但是这里只有一个类别

                
                if self.inverse_form_criterion is not None: 
                    #?  background_out 是否需要clone? 
                    #*  after detach , performance is better.
                    #* why detach?  using as 
                    if self.args.dataset == 'cityscapes':
                        background_out = pred_edge.clone().detach()
                        hard_edge_merged,__ = torch.max(hard_prediction_maps,1)
                        inverse_form_loss1 = self.inverse_form_criterion(hard_edge_merged.unsqueeze(1),background_out)
                        inverse_form_loss =  inverse_form_loss1

                    elif self.args.dataset == 'bsds':
                        background_out= output[0].clone().detach()
                        rind_out = output[1:]
                        rind_out_stack_max_value = torch.stack(rind_out).max(0)[0]
                        inverse_form_loss = self.inverse_form_criterion(rind_out_stack_max_value,background_out)

                    loss =  b_loss + rind_loss +inverse_form_loss
                else :
                    loss =  b_loss+ rind_loss  

                se_loss+=rind_loss.item()

                all_need_upload = { "val_generic_edge_loss":b_loss.item(),"val_hard_edge_loss":rind_loss.item(),"val_total_loss":loss.item()}


                if 'inverse_form_loss1' in locals():
                    all_need_upload['val_inverse_form_loss1'] = inverse_form_loss1.item()

                    
                if (i+1) % self.args.print_freq  == 0 :

                    self.wandb_log(all_need_upload)

                    tmp = 'Validation Epoch: [{0}][{1}/{2}/{3}]'.format(epoch, i, len(self.train_loader),self.args.local_rank) + \
                                "\t".join([f"{k} : {v} \t" for k,v in all_need_upload.items()])
                    self.log(tmp)

        self.last_se_edge_loss = self.current_se_edge_loss
        self.current_se_edge_loss = se_loss


    
    def validate_all_model(self):
        

        if self.args.resume_model_dir == None:
            self.log('please give the ckpt directory ')
            return 
        
        # example_dir = "/home/DISCOVER_summer2022/xusc/exp/cerberus/networks/2023-02-26-13:27:1677389259/checkpoints/ckpt_*"
        

        all_models = sorted(glob(join(self.args.resume_model_dir ,"ckpt_*")))[-10:]
        self.log(all_models)
        self.args.print_freq = 1e+10

        for model_file in tqdm(all_models):

            self.update_model(model_file)
            tic = time.time()
            self.validate_epoch(self.start_epoch)
            spend_time = time.strftime("%H:%M:%s",time.gmtime(time.time()-tic))
            
            val_loss = self.current_se_edge_loss

            self.log(f" ckpt :  {model_file.split('/')[-1]} \t val loss : {val_loss} \t spend time : {spend_time}")

            if self.last_se_edge_loss > self.current_se_edge_loss:
                self.log('current model  is better')


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
                if self.args.dataset == 'cityscapes':
                    generic_edge,indices = torch.max(target[:,1:,:,:],1)
                    pred_edge = F.sigmoid(output[0])
                    # cv2.imwrite('a.jpg',generic_edge.squeeze().cpu().numpy()*255)
                    b_loss = self.edge_criterion([pred_edge],generic_edge.unsqueeze(1))#* (B,N,W,H),(B,N,W,H)
                else:
                    b_loss = self.edge_criterion([output[0]],target[:,0,:,:].unsqueeze(1))#* (B,N,W,H),(B,N,W,H)

            elif edge_branch_out == "unet" :
                b_loss = self.focal_criterion(output[0],target[:,0,:,:])#* (B,N,W,H),(B,N,W,H)
            else :
                raise Exception('edge_branch_out is invalid:{}'.format(edge_branch_out))
            
            if self.args.dataset == 'cityscapes':
                # rind_loss = self.hard_edge_criterion(torch.cat(output[1:],dim=1),target[:,1:,:,:])#* 可以对多个类别计算loss ,但是这里只有一个类别
                hard_prediction_maps = torch.cat(output[1:],dim=1)
                rind_loss = self.hard_edge_criterion(hard_prediction_maps ,target)#* 可以对多个类别计算loss ,但是这里只有一个类别
            elif self.args.dataset == 'bsds':
                rind_loss = self.hard_edge_criterion(output[1:],target[:,1:,:,:])#* 可以对多个类别计算loss ,但是这里只有一个类别

            if torch.isnan(b_loss) or torch.isnan(rind_loss)  :
                self.log("nan")
                self.log('b_loss is: {0}'.format(b_loss))
                self.log('rind_loss is: {0}'.format(rind_loss)) 
                exit(0)

            
            if self.inverse_form_criterion is not None: 
                #?  background_out 是否需要clone? 
                #*  after detach , performance is better.
                #* why detach?  using as 
                if self.args.dataset == 'cityscapes':
                    background_out = pred_edge.clone().detach()
                    hard_edge_merged,__ = torch.max(hard_prediction_maps,1)
                    inverse_form_loss1 = self.inverse_form_criterion(hard_edge_merged.unsqueeze(1),background_out)

                    # hard_edge_merged2= torch.mean(hard_prediction_maps,1)
                    # inverse_form_loss2 = self.inverse_form_criterion(hard_edge_merged2.unsqueeze(1),background_out)
                    #?  why extra_loss == extra_loss2??
                    # inverse_form_loss = inverse_form_loss2 + inverse_form_loss1
                    inverse_form_loss =  inverse_form_loss1

                elif self.args.dataset == 'bsds':
                    background_out= output[0].clone().detach()
                    rind_out = output[1:]
                    rind_out_stack_max_value = torch.stack(rind_out).max(0)[0]
                    extra_loss = self.inverse_form_criterion(rind_out_stack_max_value,background_out)
                #* mean for merge hard task output 
                # rind_out_stack_mean_value = torch.stack(rind_out).mean(0)
                # extra_loss = inverse_form_criterion(rind_out_stack_mean_value,background_out)

                loss =  self.args.bg_weight*b_loss+ self.args.rind_weight*rind_loss \
                         + self.args.inverseform_loss_weight * inverse_form_loss
            else :
                loss =  self.args.bg_weight*b_loss+ self.args.rind_weight*rind_loss  



            self.optimizer.zero_grad()
            loss.backward()#* warning exists
            self.optimizer.step()

            #* print status 
            if  i % self.args.print_freq == 0 and self.args.local_rank == 0:#* for debug 

                all_need_upload = { "generic_edge_loss":b_loss.item(),"hard_edge_loss":rind_loss.item(),"total_loss":loss.item()}

                # locals() : 基于字典的访问局部变量的方式。键是变量名，值是变量值。
                # globals() : 基于字典的访问全局变量的方式。键是变量名，值是变量值。

                if 'inverse_form_loss1' in locals():
                    all_need_upload['inverse_form_loss1'] = inverse_form_loss1.item()

                if 'inverse_form_loss2' in locals():
                    all_need_upload['inverse_form_loss2'] = inverse_form_loss2.item()
                    
                self.wandb_log(all_need_upload)

                tmp = 'Epoch: [{0}][{1}/{2}/{3}]'.format(epoch, i, len(self.train_loader),self.args.local_rank) + \
                            "\t".join([f"{k} : {v} \t" for k,v in all_need_upload.items()])
                self.log(tmp)

    def test(self):

        self.model.eval()
        
        outdir_list_fuse = []
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
                fuse = fuses[idx]
                
                fuse =  fuse.squeeze_().sigmoid_().cpu().numpy()    
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
    


    

    