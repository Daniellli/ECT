


import os
import time
import numpy as np
import sys
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import os.path as osp
import random
import wandb
from torch.utils.data.distributed import DistributedSampler
from torchvision import  transforms
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import torch.distributed as dist
import json
from os.path import join, split, isdir, isfile, exists
from loguru import logger

from IPython import embed 

# from min_norm_solvers import MinNormSolver
#* origin model version 
from model.edge_model import EdgeCerberus


#* modified version for generalization
from model.ECT import EdgeCerberusMultiClass

from model.loss.inverse_loss import InverseTransform2D



from dataloaders.datasets.bsds_hd5 import Mydataset
from dataloaders.datasets.bsds_hd5_test import MydatasetTest


from utils import *
from utils.global_var import *
from utils.loss import SegmentationLosses
from utils.edge_option import parse_args
from utils.edge_loss2 import AttentionLoss2
# import torch.functional


import scipy.io as sio



def load_model(model_path):
    checkpoint = torch.load(model_path,map_location=torch.device('cpu'))
    return checkpoint



class ECTTrainer:


    def __init__(self,args):


        self.args = args

        #* GPU train 
        cudnn.benchmark = args.cudnn_benchmark

        #* distributed train
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)
        torch.autograd.set_detect_anomaly(True) 

        if args.local_rank == 0:
            # setting = ""
            # for k, v in args.__dict__.items():
            #     if isinstance(v,int or string or float):
            #         setting += f"#{k}{v}"
            # self.log(f"setting: {setting}")  
            if args.resume and os.path.isfile(args.resume):
                self.project_dir =  '/'.join(args.resume.split('/')[:-2])

            else :
                self.project_dir =  osp.join(osp.dirname(osp.abspath(__file__)),"networks",time.strftime("%Y-%m-%d-%H:%M:%s",time.gmtime(time.time()))+f"#CN{args.cause_token_num}")
            self.ckpt_dir=osp.join(self.project_dir,'checkpoints')
            self.log_file = join(self.project_dir,'train_log.txt')
            make_dir(self.ckpt_dir)

        if args.wandb:
            self.init_wandb()

        #* init dataloader 
        self.init_dataloader()

        #* init model 
        self.init_model()
        
        
        #* init scheduler and optimizer 
        self.edge_atten_criterion = AttentionLoss2(gamma=args.edge_loss_gamma,
                beta=args.edge_loss_beta).cuda(args.local_rank)
        self.rind_atten_criterion = AttentionLoss2(gamma=args.rind_loss_gamma,
                beta=args.rind_loss_beta).cuda(args.local_rank)
        self.focal_criterion = SegmentationLosses(weight=None, 
                cuda=True).build_loss(mode='focal')
        self.inverse_form_criterion = InverseTransform2D()
        
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                            args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)

        
        #* load model 
        if args.resume:
            if os.path.isfile(args.resume):
                checkpoint = load_model(args.resume)
                self.log("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
                self.start_epoch = checkpoint['epoch']
                self.best_prec1 = checkpoint['best_prec1']

                # if args.cmd == 'test':#* if test mode, do not use distributed model 
                #     for name, param in checkpoint['state_dict'].items():
                #         name = name.replace("module.","") 
                #         self.model.state_dict()[name].copy_(param)
                # else:
                self.model.load_state_dict(checkpoint['state_dict'], True)
                    
            else:
                self.log("=> no checkpoint found at '{}'".format(args.resume))


        if not hasattr(self,'start_epoch'):
            self.start_epoch = 0


        self.log('trainer init done ')
        self.best_val_loss = 1e+7
        self.is_best = False
        

    def log2file(self,message):
        if self.args.local_rank == 0:
            with open(self.log_file,'a')as f :
                f.write(message+"\n")

    def init_wandb(self):
        if self.args.local_rank ==0:
            wandb.init(project="train_cerberus") 
            
            for k, v in args.__dict__.items():
                if args.wandb:
                    setattr(wandb.config,k,v)
            self.use_wandb = True


    

    def init_dataloader(self):

        train_dataset = Mydataset(root_path=self.args.bsds_dir, 
            split='trainval', crop_size=args.crop_size)

        self.train_sampler = DistributedSampler(train_dataset)

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,batch_size=self.args.batch_size,
                num_workers=self.args.workers,pin_memory=True, 
                drop_last=True, sampler=self.train_sampler
        )


        # self.test_dataset = Mydataset(root_path=self.args.bsds_dir,split='test')
        self.test_dataset = MydatasetTest(root_path=self.args.bsds_dir)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, 
                        batch_size=1, shuffle=False,num_workers = self.args.workers,
                        pin_memory = False)
        self.log("Dataloader  init  done ")

        

    def init_model(self):
        #* construct model 
        # self.model = EdgeCerberus(backbone="vitb_rn50_384")
        
        self.model =  EdgeCerberusMultiClass(backbone="vitb_rn50_384",
            hard_edge_cls_num=4,cause_token_num = self.args.cause_token_num)
            
        self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model.cuda())
        self.model = torch.nn.parallel.DistributedDataParallel(self.model,device_ids=[self.args.local_rank],
                            find_unused_parameters=True,broadcast_buffers = True)
        self.log("construct model done ")



    def log(self,message):
        if self.args.local_rank == 0 :
            print(message)


    def wandb_log(self,message_dict):
        if hasattr(self,'use_wandb')  and self.use_wandb and self.args.local_rank == 0 :
            
            wandb.log(message_dict)

        
    '''
    description:  lr decay manully
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

        return lr


    def test_epoch(self,epoch):
        #! this code would lead to  extrimely low performance 
        self.log(f'start to eval {epoch}')
        

        self.model.eval()

        save_dir = join(self.project_dir,str(epoch))
        make_dir(save_dir)

            
        edge_output_dir = os.path.join(save_dir, 'all_edges/met')
        make_dir(edge_output_dir)

        depth_output_dir = os.path.join(save_dir, 'depth/met')
        make_dir(depth_output_dir)
        
        normal_output_dir = os.path.join(save_dir, 'normal/met')
        make_dir(normal_output_dir)

        reflectance_output_dir = os.path.join(save_dir, 'reflectance/met')
        make_dir(reflectance_output_dir)

        illumination_output_dir = os.path.join(save_dir, 'illumination/met')
        make_dir(illumination_output_dir)


        attention_output_dir = os.path.join(save_dir, 'attention')
        make_dir(attention_output_dir)

        tic = time.time()
        tbar = tqdm(self.test_loader, desc='\r')
        # for i, (image,__) in enumerate(tbar):#*  B,C,H,W
        for i, (image) in enumerate(tbar):#*  B,C,H,W
            if len(image) == 2 :
                image, label = image[0], image[1]


            name = self.test_loader.dataset.images_name[i] #* shuffle == false , so it sample sequentially 
            
            image = Variable(image, requires_grad=False)
            image = image.cuda()
            B,C,H,W = image.shape 
            
            # trans1 = transforms.Compose([transforms.Resize(size=(H//16*16, W//16*16))])
            trans1 = transforms.Compose([transforms.Resize(size=(H//4*4, W//4*4))])
            trans2 = transforms.Compose([transforms.Resize(size=(H, W))])
            image = trans1(image)

            # attention_save_dir = osp.join(attention_output_dir,name)
            # make_dir(attention_save_dir)
            # with open('tmp.txt' ,'w') as f :
            #     f.write(attention_save_dir)

            
            with torch.no_grad():
                
                res= self.model(image)#* out_background,out_depth, out_normal, out_reflectance, out_illumination
                
                

            out_edge = trans2(res[0])
            out_depth, out_normal, out_reflectance, out_illumination = trans2(res[1]),trans2(res[2]),trans2(res[3]),trans2(res[4])


            edge_pred = out_edge.data.cpu().numpy()
            edge_pred = edge_pred.squeeze()
            sio.savemat(os.path.join(edge_output_dir, '{}.mat'.format(name)), {'result': edge_pred})
            


            depth_pred = out_depth.data.cpu().numpy()
            depth_pred = depth_pred.squeeze()
            sio.savemat(os.path.join(depth_output_dir, '{}.mat'.format(name)), {'result': depth_pred})


            normal_pred = out_normal.data.cpu().numpy()
            normal_pred = normal_pred.squeeze()
            sio.savemat(os.path.join(normal_output_dir, '{}.mat'.format(name)), {'result': normal_pred})

            reflectance_pred = out_reflectance.data.cpu().numpy()
            reflectance_pred = reflectance_pred.squeeze()
            sio.savemat(os.path.join(reflectance_output_dir, '{}.mat'.format(name)), {'result': reflectance_pred})

            illumination_pred = out_illumination.data.cpu().numpy()
            illumination_pred = illumination_pred.squeeze()
            sio.savemat(os.path.join(illumination_output_dir, '{}.mat'.format(name)),
                        {'result': illumination_pred})
        
        #* eval by the matlab
        os.system("./eval_tools/test.sh %s %s"%(save_dir,"1"))

        
        self.log("eval done")
        with open (osp.join(save_dir,"eval_res.json"),'r')as f :
            eval_res = json.load(f)
        spend_time =  time.time() - tic
        
        self.log("spend time : "+time.strftime("%H:%M:%S",time.gmtime(spend_time)))

        return eval_res
        

    def train(self):
        
        for epoch in range(self.start_epoch,self.args.epochs):
            lr = self.adjust_learning_rate(epoch) 

            self.train_sampler.set_epoch(epoch)
            self.train_epoch(epoch)

            if epoch % self.args.save_freq == 0 and   epoch>250 :
                self.save_ckpt(epoch)
                if self.args.local_rank == 0 :
                    self.test_epoch(epoch)
                


        self.log("train finish!!!! ")



    def save_ckpt(self,epoch):
        if  self.args.local_rank == 0:
            checkpoint_path = osp.join(self.ckpt_dir,
                    'ckpt_ep%04d.pth.tar'%(epoch))
            
            
            mean_val_loss = self.validate_epoch(epoch)
            # performance = self.test_epoch(epoch)#* can not test on 10.0.0.3
            # ap = performance['Average']['AP']

            
            if self.best_val_loss > mean_val_loss:
                self.is_best = True 
                self.best_val_loss  = mean_val_loss
            else:
                self.is_best = False 
            self.log2file(f"epoch: {epoch}, val loss: {mean_val_loss}, is best {self.is_best}")
            self.wandb_log({'val_loss':mean_val_loss})
                
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'best_prec1': mean_val_loss,
            }, self.is_best, 
            filename=checkpoint_path)
        
    

    

    def validate_epoch(self,epoch):


        self.model.eval()
        loss_sum = 0

        for i, (input,target) in enumerate(self.test_loader):
            input = Variable(input,requires_grad=False).cuda()            
            target= Variable(target,requires_grad=False).cuda()

            B,C,H,W = input.shape
            trans1 = transforms.Compose([transforms.Resize(size=(H//4*4, W//4*4))])
            trans2 = transforms.Compose([transforms.Resize(size=(H, W))])
            
            input = trans1(input)
            
            with torch.no_grad():
                output = self.model(input)
            
            output = [trans2(x) for x in output]
            #* compute the loss 

            rind_loss = self.rind_atten_criterion(output[1:],target)
            # self.log(rind_loss.item())
            loss_sum+=rind_loss.item()
            if  torch.isnan(rind_loss):
                print("nan")
                self.log('rind_loss is: {0}'.format(rind_loss)) 
                exit(0)
            


            #* log
            if  i % self.args.print_freq == 0:#* for debug 
                all_need_upload = {"val_rind_loss":rind_loss}
                self.wandb_log(all_need_upload)
                tmp = 'Val Epoch: [{0}][{1}/{2}]'.format(epoch, i, len(self.test_loader))
                tmp+= "\t".join([f"{k} : {v} \t" for k,v in all_need_upload.items()])
                self.log(tmp)

        return loss_sum/len(self.test_loader)

    '''
    description: 
    criterion2 : 
    return {*}
    '''
    def train_epoch(self,epoch): # transfer_model=None, transfer_optim=None):
        
        self.model.train()

        for i, (input,target) in enumerate(self.train_loader):
            input = torch.autograd.Variable(input.cuda())
            target= torch.autograd.Variable( target.cuda())

            output = self.model(input)

            #* compute the loss 
            b_loss = self.edge_atten_criterion([output[0]],target[:,0,:,:].unsqueeze(1))#* (B,N,W,H),(B,N,W,H)          
            rind_loss = self.rind_atten_criterion(output[1:],target[:,1:,:,:])

            if torch.isnan(b_loss) or torch.isnan(rind_loss):
                print("nan")
                self.log('b_loss is: {0}'.format(b_loss))
                self.log('rind_loss is: {0}'.format(rind_loss)) 
                exit(0)

            extra_loss = self.inverse_form_criterion(
                            torch.stack(output[1:]).max(0)[0],
                            output[0].clone().detach())
            
            loss =  self.args.bg_weight*b_loss+ self.args.rind_weight*rind_loss + \
                self.args.extra_loss_weight * extra_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            #* log
            if  i % self.args.print_freq == 0:#* for debug 
                
                all_need_upload = { "b_loss":b_loss,"rind_loss":rind_loss,
                            "total_loss":loss,"extra_loss":extra_loss}
                self.wandb_log(all_need_upload)
                tmp = 'Epoch: [{0}][{1}/{2}]'.format(epoch, i, len(self.train_loader))
                tmp+= "\t".join([f"{k} : {v} \t" for k,v in all_need_upload.items()])
                self.log(tmp)

       
    

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


    
if __name__ == '__main__':
    setup_seed(20)

    args = parse_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    trainer = ECTTrainer(args)
    if args.cmd == 'train':
        trainer.train()
    elif args.cmd == 'test':
        trainer.test_epoch(trainer.start_epoch)




    

    