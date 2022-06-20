'''
Author: xushaocong
Date: 2022-06-20 22:50:51
LastEditTime: 2022-06-20 23:10:31
LastEditors: xushaocong
Description: 
FilePath: /Cerberus-main/train.py
email: xushaocong@stu.xmu.edu.cn
'''


import os
import time
import numpy as np
import sys
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from min_norm_solvers import MinNormSolver


from model.models import  CerberusSegmentationModelMultiHead
from model.edge_model import EdgeCerberus
import os.path as osp

import wandb
from loguru import logger
from utils.loss import SegmentationLosses
from utils.edge_loss2 import AttentionLoss2
from dataloaders.datasets.bsds_hd5 import Mydataset
from torch.utils.data.distributed import DistributedSampler
from utils import save_checkpoint,AverageMeter,parse_args
import json

import warnings
warnings.filterwarnings('ignore')

import torch.distributed as dist
import torch.multiprocessing as mp
import json
import signal

from utils.global_var import *

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
                input_new = in_tar_name_pair[0].clone().cuda(local_rank) 
                #* target_new  : B,W,H
                B,W,H=target_new.shape
                target_new=  target_new.reshape([1,B,1,W,H])
                #* image
                input_var_new = torch.autograd.Variable(input_new)
                
                target_var_new= [torch.autograd.Variable(target_new[idx].cuda(local_rank)) for idx in range(len(target_new))]

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
                        logger.info('loss_raw is: {0}'.format(loss_))
                        logger.info('loss_enhance is: {0}'.format(loss_enhance)) 
                        exit(0)
                    else:
                        loss_array_new.append(loss_enhance)

                task_loss_array_new.append(sum(loss_array_new))#* 只需要对一个类别计算loss , 没有其他类别不需要求和

            assert len(task_loss_array_new) == 5

            b_weight = 0.9
            rind_weight = 0.1
            
            b_loss = b_weight * task_loss_array_new[0]
            rind_loss = rind_weight*task_loss_array_new[1]+rind_weight*task_loss_array_new[2] +\
                 rind_weight*task_loss_array_new[3]+ rind_weight*task_loss_array_new[4] 

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
            else :
                all_need_upload= {}
                loss_old ={ "loss_"+k:v  for k,v  in  zip(root_task_list_array,task_loss_array_new)}
                all_need_upload.update(loss_old)
                all_need_upload.update({"total_loss":loss_new, "rind_loss":rind_loss})                
                tmp = '{3} | Epoch: [{0}][{1}/{2}]'.format(epoch, i, len(train_loader),local_rank)
                tmp+= "\t".join([f"{k} : {v} \t" for k,v in all_need_upload.items()])
                logger.info(tmp)
            
            
            optimizer.zero_grad()
            #!===========  为了解决DDP,但是解决一个问题又出现另一个问题
            # torch.autograd.set_detect_anomaly(True)
            #!=========== 
            
            loss_new.backward()
            optimizer.step()
        
        else :
            task_loss_array = []
            input_new = in_tar_name_pair[0].cuda(local_rank) 
            in_tar_name_pair_label= in_tar_name_pair[1].permute(1,0,2,3)#*  将channel 交换到第一维度
            for index,target in enumerate(in_tar_name_pair_label):# * 遍历一个一个任务
                input   = in_tar_name_pair[0].clone().cuda(local_rank)

                B,W,H=target.shape
                target=  target.reshape([1,B,1,W,H])
                # measure data loading time #* input 输入图像数据, target 图像的标签, name : input 的相对路径
                data_time_list[index].update(time.time() - end)
                '''
                #! 主要就是将数据前向推理 然后计算loss ,存储loss,  然后是反向传播,处理梯度 ?, 不是很清楚是如何处理梯度的  
                '''
                input_var = torch.autograd.Variable(input)#?将输入转成可导的对象 
                target_var = [torch.autograd.Variable(target[idx].cuda(local_rank)) for idx in range(len(target)) ]
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
                        input_new=in_tar_name_pair[0].clone().cuda(local_rank)
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
description:  修改 lock 的互斥信号量
'''
lock = False
'''
description:  读取节点数,这个数据
return {*}
'''
def read_ddp():
    global lock
    while lock:
        pass

    with open(ddp_file,'r')as f :
        data  = json.load(f)
    
    return data

''' 
description:  写入节点数, 互斥, 一次只能一个进程写入
param {*} data
return {*}
'''
def write_ddp(data):
    global lock
    while lock:
        pass
    lock = True
    with open(ddp_file,'w') as f :
        json.dump(data,f)
    lock = False
        
'''
description: 平行节点数-1的操作, 
return {*}
'''
def minus_process():
    data = read_ddp()
    data["node"] -=1
    write_ddp(data)
    logger.info(f"minus process ,surplus : {data['node']}")
    return data["node"]





'''
description:  
param {*} local_rank : 分布式训练参数 , 考试当前第几个节点,第几个GPU
param {*} nprocs:  分布式训练参数 , 表示共有几个节点用于计算每个节点的batch size
param {*} args : 训练脚本的超参数
return {*}
'''
def train_seg_cerberus(local_rank,nprocs,  args):
    
    logger.info(f"local_rank: {local_rank},nprocs={nprocs}")#* local_rank : int type
    args.local_rank = local_rank
    
    # port = int(1e4+np.random.randint(1,10000,1)[0])
    dist.init_process_group(backend='nccl',
                        # init_method='tcp://127.0.0.1:%d'%(port),
                        world_size=args.nprocs,
                        rank=local_rank)

    torch.cuda.set_device(args.local_rank) 
    logger.info("DDP init  done ")
    
    model_save_dir = None
    if args.local_rank == 0: 
        run = wandb.init(project="train_cerberus") 
        

        run.name+= "_lr@%s_ep@%s"%(args.lr,args.epochs,)#* 改名

        args.project_name = run.name 
        info =""
        for k, v in args.__dict__.items():
            setattr(wandb.config,k,v)
            info+= ( str(k)+' : '+ str(v))
        logger.info(info)
        logger.info(' '.join(sys.argv))

        model_save_dir =  osp.join("networks",args.project_name,"checkpoints")
        if not osp.exists(model_save_dir):
            os.makedirs(model_save_dir)

    
    #* construct model 
    # single_model = CerberusSegmentationModelMultiHead(backbone="vitb_rn50_384")
    single_model = EdgeCerberus(backbone="vitb_rn50_384")

    
    model = single_model.cuda(local_rank)
    # model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[args.local_rank]) #* 问题一大推, 
    # model = torch.nn.parallel.DistributedDataParallel(model) #* 还是会出错, default setting 不行
    model = torch.nn.DataParallel(model,device_ids=[args.local_rank])
    
    logger.info("construct model done ")
    cudnn.benchmark = args.cudnn_benchmark
    #*=====================================
    atten_criterion = AttentionLoss2().cuda(local_rank)
    focal_criterion = SegmentationLosses(weight=None, cuda=True).build_loss(mode='focal')



    #* 不能整除怎么办
    if (args.batch_size % args.nprocs) != 0 and local_rank==0 :
        args.batch_size  = int(args.batch_size / args.nprocs)  + args.batch_size % args.nprocs #* 不能整除的部分加到第一个GPU
    else :
        args.batch_size = int(args.batch_size / args.nprocs)

    #* Data loading code
    logger.info(f"rank = {local_rank},batch_size == {args.batch_size}")
    train_dataset = Mydataset(root_path=args.train_dir, split='trainval', crop_size=args.crop_size)
    train_sampler = DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,batch_size=args.batch_size, num_workers=args.workers,
            pin_memory=True, drop_last=True, sampler=train_sampler
    )

    logger.info("Dataloader  init  done ")

    #* load test data =====================
    # test_dataset = Mydataset(root_path=args.test_dir, split='test', crop_size=args.crop_size)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, 
    #                         shuffle=False,num_workers=args.workers,pin_memory=False)
    #*=====================================

    # define loss function (criterion) and pptimizer
    optimizer = torch.optim.SGD([
                                {'params':single_model.pretrained.parameters()},
                                {'params':single_model.scratch.parameters()}],
                                # {'params':single_model.sigma.parameters(), 'lr': args.lr * 0.01}],
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
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

        #*=====================
        train_sampler.set_epoch(epoch)
        #*=====================
        train_cerberus(train_loader, model, atten_criterion,
             focal_criterion,optimizer, epoch,_moo = args.moo,local_rank = args.local_rank,print_freq=1)
        #if epoch%10==1:
        # prec1 = validate_cerberus(val_loader, model, criterion, eval_score=mIoU, epoch=epoch)
        # wandb.log({"prec":prec1})
        # is_best = prec1 > best_prec1
        # best_prec1 = max(prec1, best_prec1)
        if args.local_rank == 0 :
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
    logger.info("train finish!!!! ")
    logger.info(f"{os.getppid()} exit !!! ")
    surplus_process = minus_process()
    if surplus_process == 0 :
        logger.info(f"ready to kill ")
        # os.kill(os.getpid(),signal.SIGKILL) #*  did not work 
        wandb.finish(0)
        os.kill(os.getppid(),signal.SIGKILL)
        
    
    




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



def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    port = int(1e4+np.random.randint(1,10000,1)[0])
    logger.info(f"port == {port}")
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    # train_seg_cerberus(args)
    #* 分布式training
    args.nprocs = torch.cuda.device_count()#* gpu  number 

    write_ddp({"node": args.nprocs})
    logger.info(f"node number == {args.nprocs}")
    mp.spawn(train_seg_cerberus,nprocs=args.nprocs, args=(args.nprocs, args))
    # mp.spawn(train_seg_cerberus,nprocs=args.nprocs, args=(args.nprocs, args),join=True)#* 加了这个join ,进程之间就会相互等待
    
 
if __name__ == '__main__':
    main()

    