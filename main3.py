'''
Author: xushaocong
Date: 2022-06-07 19:13:11
LastEditTime: 2022-06-16 17:09:59
LastEditors: xushaocong
Description:  改成5个头部
FilePath: /Cerberus-main/main3.py
email: xushaocong@stu.xmu.edu.cn
'''
#!/usr/bin/env python
# -*- coding: utf-8 -*-


import imp
import os
# from IPython import embed #for terminal debug 
# import pdb
import time
import numpy as np
import sys
import torch
import torch.backends.cudnn as cudnn
from torchvision import  transforms
from torch.autograd import Variable
# from tensorboardX import SummaryWriter
from min_norm_solvers import MinNormSolver

import data_transforms as transforms
from model.models import DPTSegmentationModel, DPTSegmentationModelMultiHead, TransferNet, CerberusSegmentationModelMultiHead
# from model.transforms import PrepareForNet
import os.path as osp
from tqdm import tqdm
import scipy.io as sio
import torchvision.transforms as transforms
# from torchcam.methods import SmoothGradCAMpp
# import torch.nn.functional as F

#*===================
import glob
import wandb
from loguru import logger
from utils.loss import SegmentationLosses
from utils.edge_loss2 import AttentionLoss2
from dataloaders.datasets.bsds_hd5 import Mydataset
from torch.utils.data.distributed import DistributedSampler
from utils import accuracy,downsampling,fill_up_weights,\
    save_colorful_images,resize_4d_tensor,make_dir,fast_hist,\
        save_checkpoint,AverageMeter,parse_args
import json

# sys.path.append(osp.join(osp.dirname(__file__),"eval_tools")) #* 加这行无效!
# sys.path.append(osp.join(osp.dirname(__file__),"eval_tools","edges")) #* 加这行无效!
# sys.path.append(osp.join(osp.dirname(__file__),"eval_tools","edges","private2")) #* 加这行无效!
# from eval_tools.test import test_by_matlab
import warnings
warnings.filterwarnings('ignore')



#*====================
TASK =None  # 'ATTRIBUTE', 'AFFORDANCE', 'SEGMENTATION' 
TRANSFER_FROM_TASK = None  #'ATTRIBUTE', 'AFFORDANCE', 'SEGMENTATION', or None to unable transfer

CITYSCAPE_PALETTE = np.asarray([
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [70, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32],
    [0, 0, 0]], dtype=np.uint8)

NYU40_PALETTE = np.asarray([
    [0, 0, 0], 
    [0, 0, 80], 
    [0, 0, 160], 
    [0, 0, 240], 
    [0, 80, 0], 
    [0, 80, 80], 
    [0, 80, 160], 
    [0, 80, 240], 
    [0, 160, 0], 
    [0, 160, 80], 
    [0, 160, 160], 
    [0, 160, 240], 
    [0, 240, 0], 
    [0, 240, 80], 
    [0, 240, 160], 
    [0, 240, 240], 
    [80, 0, 0], 
    [80, 0, 80], 
    [80, 0, 160], 
    [80, 0, 240], 
    [80, 80, 0], 
    [80, 80, 80], 
    [80, 80, 160], 
    [80, 80, 240], 
    [80, 160, 0], 
    [80, 160, 80], 
    [80, 160, 160], 
    [80, 160, 240], [80, 240, 0], [80, 240, 80], [80, 240, 160], [80, 240, 240], 
    [160, 0, 0], [160, 0, 80], [160, 0, 160], [160, 0, 240], [160, 80, 0], 
    [160, 80, 80], [160, 80, 160], [160, 80, 240]], dtype=np.uint8)


AFFORDANCE_PALETTE = np.asarray([
    [0, 0, 0],
    [255, 255, 255]], dtype=np.uint8)


task_list = None
middle_task_list = None

if TASK == 'ATTRIBUTE':
    task_list = ['Wood','Painted','Paper','Glass','Brick','Metal','Flat','Plastic','Textured','Glossy','Shiny']
    FILE_DESCRIPTION = '_attribute'
    PALETTE = AFFORDANCE_PALETTE
    EVAL_METHOD = 'mIoU'
elif TASK == 'AFFORDANCE':
    task_list = ['L','M','R','S','W']
    FILE_DESCRIPTION = '_affordance'
    PALETTE = AFFORDANCE_PALETTE
    EVAL_METHOD = 'mIoU'
elif TASK =='SEGMENTATION':
    task_list = ['Segmentation']
    FILE_DESCRIPTION = ''
    PALETTE = NYU40_PALETTE
    EVAL_METHOD = 'mIoUAll'
else:
    task_list = None
    FILE_DESCRIPTION = ''
    PALETTE = None
    EVAL_METHOD = None

if TRANSFER_FROM_TASK == 'ATTRIBUTE':
    middle_task_list = ['Wood','Painted','Paper','Glass','Brick','Metal','Flat','Plastic','Textured','Glossy','Shiny']
elif TRANSFER_FROM_TASK == 'AFFORDANCE':
    middle_task_list = ['L','M','R','S','W']
elif TRANSFER_FROM_TASK =='SEGMENTATION':
    middle_task_list = ['Segmentation']
elif TRANSFER_FROM_TASK is None:
    pass


# if TRANSFER_FROM_TASK is not None:
#     TENSORBOARD_WRITER = SummaryWriter(comment='From_'+TRANSFER_FROM_TASK+'_TO_'+TASK)
# elif TASK is not None:
#     TENSORBOARD_WRITER = SummaryWriter(comment=TASK)
# else:
#     TENSORBOARD_WRITER = SummaryWriter(comment='Nontype')



'''
description: 
criterion2 : 
return {*}
'''
def train_cerberus(train_loader, model, atten_criterion,focal_criterion ,optimizer, epoch,
          eval_score=None, print_freq=1,_moo=False,local_rank=0): # transfer_model=None, transfer_optim=None):
    
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

    model.train()
    end = time.time()

    moo = _moo
    for i, in_tar_name_pair in enumerate(train_loader):#* 一个一个batch取数据
        if moo :
            grads = {}
        
        if not moo:
            task_loss_array_new=[]
            in_tar_name_pair_label= in_tar_name_pair[1].permute(1,0,2,3)#*  将channel 交换到第一维度
            for index_new, (target_new) in enumerate(in_tar_name_pair_label):
                input_new = in_tar_name_pair[0].clone().cuda() 
                #* target_new  : B,W,H
                B,W,H=target_new.shape
                target_new=  target_new.reshape([1,B,1,W,H])
                #* image
                input_var_new = torch.autograd.Variable(input_new)
                
                target_var_new= [torch.autograd.Variable(target_new[idx].cuda()) for idx in range(len(target_new))]

                output_new, _, _ = model(input_var_new, index_new)
                loss_array_new = list()
                for idx in range(len(output_new)):
                    if index_new == 0:
                        loss_=focal_criterion(output_new[idx],target_var_new[idx][:,0,:,:])#* (B,N,W,H),(B,N,W,H)
                    else:
                        loss_=atten_criterion(output_new[idx],target_var_new[idx][:,0:1,:,:])#* 可以对多个类别计算loss ,但是这里只有一个类别
                    loss_enhance = loss_
                    if torch.isnan(loss_enhance):
                        print("nan")
                        logger.info('loss_raw is: {0}'.format(loss_raw))
                        logger.info('loss_enhance is: {0}'.format(loss_enhance)) 
                        exit(0)
                    else:
                        loss_array_new.append(loss_enhance)

                task_loss_array_new.append(sum(loss_array_new))#* 只需要对一个类别计算loss , 没有其他类别不需要求和

            assert len(task_loss_array_new) == 5

            b_loss = 0.9 * task_loss_array_new[0]
            rind_loss = 0.1*task_loss_array_new[1]+0.1*task_loss_array_new[2] +\
                 0.1*task_loss_array_new[3]+ 0.1*task_loss_array_new[4] 
            loss_new =  b_loss+ rind_loss

            if  i % print_freq == 0 and local_rank ==0:
                all_need_upload= {}
                loss_old ={ "loss_"+k:v  for k,v  in  zip(root_task_list_array,task_loss_array_new)}
                all_need_upload.update(loss_old)
                all_need_upload.update({"total_loss":loss_new, "rind_loss":rind_loss})
                wandb.log(all_need_upload)
                tmp = 'Epoch: [{0}][{1}/{2}]'.format(epoch, i, len(train_loader))
                tmp+= "\t".join([f"{k} : {v} \t" for k,v in all_need_upload.items()])

                logger.info(tmp)

            optimizer.zero_grad()
            loss_new.backward()
            optimizer.step()
        
        else :
            task_loss_array = []
            input_new = in_tar_name_pair[0].cuda() 
            in_tar_name_pair_label= in_tar_name_pair[1].permute(1,0,2,3)#*  将channel 交换到第一维度
            for index,target in enumerate(in_tar_name_pair_label):# * 遍历一个一个任务
                input   = in_tar_name_pair[0].clone().cuda()

                B,W,H=target.shape
                target=  target.reshape([1,B,1,W,H])
                # measure data loading time #* input 输入图像数据, target 图像的标签, name : input 的相对路径
                data_time_list[index].update(time.time() - end)
                '''
                #! 主要就是将数据前向推理 然后计算loss ,存储loss,  然后是反向传播,处理梯度 ?, 不是很清楚是如何处理梯度的  
                '''
                input_var = torch.autograd.Variable(input)#?将输入转成可导的对象 
                target_var = [torch.autograd.Variable(target[idx].cuda()) for idx in range(len(target)) ]
                # compute output
                output, _, _ = model(input_var, index)
                loss_array = list()
                #! 之前的任务是 每个类计算loss, 以[11,1,2,512,512]  为例子, 会循环11次, 输入[2,512,512] 输出[1,512,512] 也就是每个像素是否属于这个像素, 然后和gt[1,512,512]计算loss
                for idx in range(len(output)):#? 遍历该子任务预测的各个类别mask, 然后我只有一个类别, 并且只输出概率
                    if index==0:
                        loss_raw = focal_criterion(output[idx],target_var[idx][:,0,:,:]) #* 计算loss, output 经过softmax 变成0-1 ,但是target_var 确实0-255
                    else :
                        loss_raw = atten_criterion(output[idx],target_var[idx][:,0:1,:,:]) #* 计算loss, output 经过softmax 变成0-1 ,但是target_var 确实0-255

                    loss_enhance = loss_raw 
                    if torch.isnan(loss_enhance):
                        print("nan")
                        logger.info('loss_raw is: {0}'.format(loss_raw))
                        logger.info('loss_enhance is: {0}'.format(loss_enhance)) 
                        exit(0)
                    else:
                        loss_array.append(loss_enhance)
                    
                local_loss = sum(loss_array)
                local_loss_enhance = local_loss 
                
                # backward for gradient calculate
                for cnt in model.pretrained.parameters():
                    cnt.grad = None
                model.scratch.layer1_rn.weight.grad = None
                model.scratch.layer2_rn.weight.grad = None
                model.scratch.layer3_rn.weight.grad = None
                model.scratch.layer4_rn.weight.grad = None
                local_loss_enhance.backward() #? 反向传播

                grads[root_task_list_array[index]] = []
                #? 遍历backbone 的参数是要干嘛?  存储梯度? 
                for par_name, cnt in model.pretrained.named_parameters():
                    if cnt.grad is not None:
                        grads[root_task_list_array[index]].append(Variable(cnt.grad.data.clone(),requires_grad = False))
                grads[root_task_list_array[index]].append(Variable(model.scratch.layer1_rn.weight.grad.data.clone(), requires_grad = False))
                grads[root_task_list_array[index]].append(Variable(model.scratch.layer2_rn.weight.grad.data.clone(), requires_grad = False))
                grads[root_task_list_array[index]].append(Variable(model.scratch.layer3_rn.weight.grad.data.clone(), requires_grad = False))
                grads[root_task_list_array[index]].append(Variable(model.scratch.layer4_rn.weight.grad.data.clone(), requires_grad = False))
                
        
                # measure accuracy and record loss
                if torch.isnan(local_loss_enhance):
                    print("nan")
                    logger.info('loss_raw is: {0}'.format(local_loss))
                    logger.info('loss_enhance is: {0}'.format(local_loss_enhance))
                    exit(0)
                else:
                    task_loss_array.append(local_loss_enhance)

                losses_list[index].update(local_loss_enhance.item(), input.size(0))#* losses_list记录loss 
                for idx, it in enumerate(task_list_array[index]): #* 遍历当前处理的子任务的每个类别, 每个类别都有一个loss ,记录到losses_array_list
                    (losses_array_list[index][idx]).update((loss_array[idx]).item(), input.size(0))

                # compute gradient and do SGD step
                if index == 4:           
                    '''
                    description:  在最后一个子任务 , 前向推理结束
                    #! 将所有子任务在重新前向推理一次, 计算loss , 对应文章的二次求导
                    '''
                    del input, target, input_var, target_var
                    task_loss_array_new = []
                    #! 重新前向传播计算梯度
                    for index_new, target_new in enumerate(in_tar_name_pair_label):
                        B,W,H=target_new.shape
                        input_new=in_tar_name_pair[0].clone().cuda()
                        target_new=  target_new.reshape([1,B,1,W,H])
                        input_var_new = torch.autograd.Variable(input_new)
                        target_var_new= [torch.autograd.Variable(target_new[idx].cuda()) for idx in range (len(target_new)) ]

                        output_new, _, _ = model(input_var_new, index_new)
                        
                        loss_array_new=list()
                        for idx in range(len(output_new)):
                            if index_new==0:    
                                loss_ = focal_criterion(output_new[idx],target_var_new[idx][:,0,:,:] )
                            else :
                                loss_ = atten_criterion(output_new[idx],target_var_new[idx][:,0:1,:,:]) 

                            if torch.isnan(loss_):
                                print("nan")
                                logger.info('loss_raw is: {0}'.format(loss_raw))
                                logger.info('loss_enhance is: {0}'.format(loss_enhance)) 
                                exit(0)
                            else:
                                loss_array_new.append(loss_)


                        task_loss_array_new.append(sum(loss_array_new))

                    assert len(task_loss_array_new) == 5

                    sol, min_norm = MinNormSolver.find_min_norm_element([grads[cnt] for cnt in root_task_list_array])
                    #* 打印一下
                    scale_info ='scale is: |{0}|\t|{1}|\t|{2}|\t {3} \t {4}'.format(sol[0], sol[1], sol[2],sol[3],sol[4]) 
                    logger.info(scale_info)
                    # print(scale_info)

                    b_loss =  sol[0] * task_loss_array_new[0]
                    rind_loss =  sol[1] * task_loss_array_new[1] + sol[2] * task_loss_array_new[2] +  \
                            sol[3] * task_loss_array_new[3]  +sol[4] * task_loss_array_new[4] 
                    loss_new = b_loss+ rind_loss

                        
                    all_need_upload= {}
                    loss_old ={ "loss_"+k:v  for k,v  in  zip(root_task_list_array,task_loss_array_new)}
                    # loss_scale= { "scale_"+k:v  for k,v  in  zip(root_task_list_array,sol)}
                    # all_need_upload.update(loss_scale)
                    all_need_upload.update(loss_old)
                    all_need_upload.update({"total_loss_with_scale":loss_new,"rind_loss":rind_loss})
                    if   i % print_freq == 0 and local_rank== 0:
                        wandb.log(all_need_upload)
                        logger.info("\t".join([f"{k}:{v} " for k,v in all_need_upload.items()]))
                        
                    #*print loss info 
                    optimizer.zero_grad()
                    loss_new.backward()
                    optimizer.step()

            # measure elapsed time
            batch_time_list[index].update(time.time() - end)
            end = time.time()
            if i % print_freq == 0:
                losses_info = ''
                for idx, it in enumerate(task_list_array[index]):
                    losses_info += 'Loss_{0} {loss.val:.4f} ({loss.avg:.4f}) \t'.format(it, loss=losses_array_list[index][idx])
                logger.info('Epoch: [{0}][{1}/{2}]\t'
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                            '{loss_info}'.format(
                    epoch, i, len(train_loader), batch_time=batch_time_list[index],
                    data_time=data_time_list[index], loss=losses_list[index],loss_info=losses_info))
                 



'''
description: 
param {*} args
param {*} config : wandb config
param {*} run: wandb run 
return {*}
'''
def train_seg_cerberus(args):

    model_save_dir = None
    print(f"os local rank = {os.environ['LOCAL_RANK']}")
    if os.environ['LOCAL_RANK'] == 0 or not args.distributed_train: 
        run = wandb.init(project="train_cerberus") 
        project_name = run.name
        info =""
        for k, v in args.__dict__.items():
            setattr(wandb.config,k,v)
            info+= ( str(k)+' : '+ str(v))
        logger.info(info)
        logger.info(' '.join(sys.argv))

        model_save_dir =  osp.join("networks",project_name,"checkpoints")
        if not osp.exists(model_save_dir):
            os.makedirs(model_save_dir)

    model =single_model=  None
    print(f"local_rank = {os.environ['LOCAL_RANK']}")
    #* 分布式
    if args.distributed_train:
        torch.cuda.set_device(os.environ['LOCAL_RANK']) 
        torch.distributed.init_process_group(backend='nccl')
        single_model = CerberusSegmentationModelMultiHead(backbone="vitb_rn50_384")
        model = single_model.cuda()
<<<<<<< HEAD
        # model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[args.local_rank],
        #         output_device=args.local_rank) #*output_device 是最终将数据汇总到哪里
        model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[args.local_rank],output_device=0)
=======
        model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[os.environ['LOCAL_RANK']],
                output_device=os.environ['LOCAL_RANK']) 
>>>>>>> 13f0a1f3ebcd556d62bf6f5ad266df86d73de7b6
        model=model.module
    else :
        single_model = CerberusSegmentationModelMultiHead(backbone="vitb_rn50_384")
        model = single_model.cuda()

    atten_criterion = AttentionLoss2().cuda()
    focal_criterion = SegmentationLosses(weight=None, cuda=True).build_loss(mode='focal')
    # Data loading code
    train_dataset = Mydataset(root_path=args.train_dir, split='trainval', crop_size=args.crop_size)

    #* 分布式训练
    train_sampler = None 
    if args.distributed_train : 
        train_sampler = DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,batch_size=args.batch_size, num_workers=args.workers,
            pin_memory=True, drop_last=True, sampler=train_sampler
    )

    #* load test data =====================
    test_dataset = Mydataset(root_path=args.test_dir, split='test', crop_size=args.crop_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, 
                            shuffle=False,num_workers=args.workers,pin_memory=False)
    
    #*=====================================

    # define loss function (criterion) and pptimizer
    optimizer = torch.optim.SGD([
                                {'params':single_model.pretrained.parameters()},
                                {'params':single_model.scratch.parameters()}],
                                # {'params':single_model.sigma.parameters(), 'lr': args.lr * 0.01}],
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    cudnn.benchmark = True
    best_prec1 = 0
    start_epoch = 0

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            
            for name, param in checkpoint['state_dict'].items():
                # name = name[7:]
                model.state_dict()[name].copy_(param)
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    elif args.pretrained_model:
        if os.path.isfile(args.pretrained_model):
            logger.info("=> loading pretrained checkpoint '{}'".format(args.pretrained_model))
            checkpoint = torch.load(args.pretrained_model)
            for name, param in checkpoint['state_dict'].items():
                if name[:5] == 'sigma':
                    pass
                    # model.state_dict()[name].copy_(param)
                else:
                    model.state_dict()[name].copy_(param)
                    # pass
            logger.info("=> loaded model checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))
    


    for epoch in range(start_epoch, args.epochs):

        lr = adjust_learning_rate(args, optimizer, epoch)
        logger.info('Epoch: [{0}]\tlr {1:.06f}'.format(epoch, lr))
        train_cerberus(train_loader, model, atten_criterion,
             focal_criterion,optimizer, epoch,_moo = args.moo,local_rank = os.environ['LOCAL_RANK'])
        #if epoch%10==1:
        # prec1 = validate_cerberus(val_loader, model, criterion, eval_score=mIoU, epoch=epoch)
        # wandb.log({"prec":prec1})
        # is_best = prec1 > best_prec1
        # best_prec1 = max(prec1, best_prec1)
        if os.environ['LOCAL_RANK'] == 0  or not args.distributed_train:
            is_best =True #* 假设每次都是最好的 
            checkpoint_path = osp.join(model_save_dir,'checkpoint_ep%04d.pth.tar'%epoch)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=checkpoint_path)

        #* test in last epoch 
        #* can not be test in 10.0.0.254 
        # if epoch +1 == args.epochs:
        #     wandb.log(test_edge(osp.abspath(checkpoint_path),test_loader))
        # if (epoch + 1) % 5 == 0:
        #     history_path =osp.join("networks/exp1",'checkpoint_{:03d}.pth.tar'.format(epoch + 1)) 
        #     shutil.copyfile(checkpoint_path, history_path)


def adjust_learning_rate(args, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if args.lr_mode == 'step':
        lr = args.lr * (0.1 ** (epoch // args.step))
    elif args.lr_mode == 'poly':
        lr = args.lr * (1 - epoch / args.epochs) ** 0.9
    else:
        raise ValueError('Unknown lr mode {}'.format(args.lr_mode))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    #adjust the learning rate of sigma
    optimizer.param_groups[-1]['lr'] = lr * 0.01
    
    return lr




'''
description: 
param {*} args
return {*}
'''
def test_edge(model_abs_path,test_loader,runid=None ):
    tic = time.time()
    
    a = osp.split(model_abs_path)
    if runid is  None:
        output_dir  = osp.join(a[0],"..","model_res")
    else:
        output_dir  = osp.join(a[0],"..","model_res_%d"%runid)

    # output_dir = osp.join("/".join(args.resume.split("/")[:-2]),"model_res")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # logger.info(output_dir)
    
    
    #* 加载模型
    single_model = CerberusSegmentationModelMultiHead(backbone="vitb_rn50_384")
    checkpoint = torch.load(model_abs_path,map_location='cuda:0')
    for name, param in checkpoint['state_dict'].items():
        single_model.state_dict()[name].copy_(param)
    model = single_model.cuda()
    model.eval()
    
  

    cudnn.benchmark = True
    depth_output_dir = os.path.join(output_dir, 'depth/met')
    make_dir(depth_output_dir)
    
    normal_output_dir = os.path.join(output_dir, 'normal/met')
    make_dir(normal_output_dir)

    reflectance_output_dir = os.path.join(output_dir, 'reflectance/met')
    make_dir(reflectance_output_dir)

    illumination_output_dir = os.path.join(output_dir, 'illumination/met')
    make_dir(illumination_output_dir)

    #* 判断一些是否测试过了 , 测试过就不重复测试了
    if not(len(glob.glob(normal_output_dir+"/*.mat")) == len(test_loader)): 
        model.eval()
        tbar = tqdm(test_loader, desc='\r')
        for i, image in enumerate(tbar):#*  B,C,H,W
            name = test_loader.dataset.images_name[i]
            image = Variable(image, requires_grad=False)
            image = image.cuda()
            B,C,H,W = image.shape 
            trans1 = transforms.Compose([transforms.Resize(size=(320, 480))])
            trans2 = transforms.Compose([transforms.Resize(size=(H, W))])
            image = trans1(image)#* debug

            with torch.no_grad():
                res = list()#* out_depth, out_normal, out_reflectance, out_illumination
                for idx in range(0,5,1):#* 第0个分支不用推理, 
                    tmp,_,_ = model(image,idx)
                    res.append(trans2(tmp[0])) #* debug


            out_depth, out_normal, out_reflectance, out_illumination = res[1],res[2],res[3],res[4]
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
    #* 因为环境冲突, 用另一个shell激活另一个虚拟环境, 进行eval
    os.system("./eval_tools/test.sh %s"%output_dir)
    #* 读取评估的结果
    with open (osp.join(output_dir,"eval_res.json"),'r')as f :
        eval_res = json.load(f)

    spend_time =  time.time() - tic
    #* 计算耗时
    logger.info("spend time : "+time.strftime("%H:%M:%S",time.gmtime(spend_time)))

    return eval_res
    

def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    if args.cmd == 'train':
       train_seg_cerberus(args)
    elif args.cmd == 'test':
        #* load data 
        train_dataset = Mydataset(root_path=args.test_dir, split='test', crop_size=args.crop_size)
        test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, 
                            shuffle=False,num_workers=args.workers,pin_memory=False)
        test_edge(args.resume,test_loader,args.run_id)#! resume 给的model path需要是绝对路径
if __name__ == '__main__':
    main()
    torch.cuda.empty_cache()
