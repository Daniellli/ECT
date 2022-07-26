


import os
import time
from cv2 import threshold
import numpy as np
import sys
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from min_norm_solvers import MinNormSolver

import copy
# from model.models import  CerberusSegmentationModelMultiHead
from model.edge_model import EdgeCerberus
import os.path as osp


import random
import wandb
from loguru import logger
from utils.loss import SegmentationLosses
from utils.edge_loss2 import AttentionLoss2
from dataloaders.datasets.bsds_hd5 import Mydataset
from torch.utils.data.distributed import DistributedSampler
from utils import save_checkpoint,AverageMeter,parse_args,calculate_param_num
import json

import warnings
warnings.filterwarnings('ignore')

import torch.distributed as dist
import torch.multiprocessing as mp
import json
import signal

from utils.global_var import *


from utils.check_model_consistent import is_model_consistent

from model.loss.inverse_loss import InverseTransform2D


from test import edge_validation

'''
description: 
criterion2 : 
return {*}
'''
def train_cerberus(train_loader, model, atten_criterion,focal_criterion ,optimizer, epoch,
          eval_score=None, print_freq=1,_moo=False,local_rank=0,bg_weight=1,rind_weight=1,
          extra_loss_weight=0.1,inverse_form_criterion  = None): # transfer_model=None, transfer_optim=None):
    
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
        
            input = torch.autograd.Variable(in_tar_name_pair[0].cuda(local_rank) )
            target= torch.autograd.Variable( in_tar_name_pair[1].cuda(local_rank))
            output = model(input)

            # background_data  = target[:,0,:,:][1][target[:,0,:,:][1]==0].shape[0] 
            # edge_data = target[:,0,:,:][1][target[:,0,:,:][1]==1].shape[0] + \
            #             target[:,0,:,:][1][target[:,0,:,:][1]==2].shape[0] + \
            #             target[:,0,:,:][1][target[:,0,:,:][1]==3].shape[0]+\
            #             target[:,0,:,:][1][target[:,0,:,:][1]==4].shape[0]+\
            #             target[:,0,:,:][1][target[:,0,:,:][1]==255].shape[0]+\

            # b_loss=focal_criterion(output[0],target[:,0,:,:])#* (B,N,W,H),(B,N,W,H)
            
            
            
            b_loss=atten_criterion([output[0]],target[:,0,:,:].unsqueeze(1))#* (B,N,W,H),(B,N,W,H)
            rind_loss=atten_criterion(output[1:],target[:,1:,:,:])#* 可以对多个类别计算loss ,但是这里只有一个类别


            #!+======================================================
            # rind_threshold =0.5
            #?  background_out 是否需要clone? 
            #*  after detach , performance is better.
            background_out= output[0].clone().detach()
            rind_out = output[1:]
            # rind_out_stack_max_value = torch.stack(rind_out).max(0)[0]
            # extra_loss = inverse_form_criterion(rind_out_stack_max_value,background_out)
            rind_out_stack_mean_value = torch.stack(rind_out).mean(0)
            extra_loss = inverse_form_criterion(rind_out_stack_mean_value,background_out)
            
            
            #* can not  use it to constrain four subtask if not  using threshold to map 
            #todo: use  different threshold to map edge detection output 
            # background_out[background_out >=rind_threshold ] = 1
            # background_out[background_out <rind_threshold ] = 0
            
            # rind_out_stack_max_value = tmp[0]
            # rind_out_stack_max_indices = tmp[1]
            
            # tmp = torch.zeros(rind_out[0].shape).bool().to(rind_out[0].device)
            # for t in rind_out:
            #     t = t.clone()#* will lead to gradient error if not clone,  
            #     #* can not combine four map  if not  using threshold to map 
            #     t[t >=rind_threshold ] = 1
            #     t[t <rind_threshold ] = 0
            #     tmp =  tmp | t.bool()
            # extra_loss = atten_criterion([background_out] ,tmp.float())
            # extra_loss = atten_criterion([tmp.float()] ,background_out)
            # extra_loss = atten_criterion([rind_out_stack_max_value] ,background_out)
            #!+======================================================
            
            if torch.isnan(b_loss) or torch.isnan(rind_loss)  :
                print("nan")
                logger.info('b_loss is: {0}'.format(b_loss))
                logger.info('rind_loss is: {0}'.format(rind_loss)) 
                exit(0)

            
            loss =  bg_weight*b_loss+ rind_weight*rind_loss   + extra_loss_weight * extra_loss


            # if  i % print_freq == 0 and local_rank ==0:
            if  i % print_freq == 0 and local_rank == 0:#* for debug 
                all_need_upload = { "b_loss":b_loss,"rind_loss":rind_loss,"total_loss":loss,"extra_loss":extra_loss}
                wandb.log(all_need_upload)
                tmp = 'Epoch: [{0}][{1}/{2}/{3}]'.format(epoch, i, len(train_loader),local_rank)
                tmp+= "\t".join([f"{k} : {v} \t" for k,v in all_need_upload.items()])
                logger.info(tmp)

                
            optimizer.zero_grad()
            loss.backward()
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
def train_seg_cerberus(args):
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank) 
    logger.info("DDP init  done ")
    
    model_save_dir = None

    if args.local_rank == 0 : 
        wandb.init(project="train_cerberus") 
        model_save_dir = args.save_dir
        logger.info(f"bg_weight = {args.bg_weight},rind_weight = {args.rind_weight} ")
        
        info =""
        for k, v in args.__dict__.items():
            setattr(wandb.config,k,v)
            info+= ( str(k)+' : '+ str(v))
        setattr(wandb.config,"extra_loss_weight",args.extra_loss_weight)
        if not osp.exists(model_save_dir):
            os.makedirs(model_save_dir)

    
    #* construct model 
    # single_model = CerberusSegmentationModelMultiHead(backbone="vitb_rn50_384")
    single_model = EdgeCerberus(backbone="vitb_rn50_384")

    #*========================================================
    #* calc parameter numbers 
    total_params,Trainable_params,NonTrainable_params =calculate_param_num(single_model)
    logger.info(f"total_params={total_params},Trainable_params={Trainable_params},NonTrainable_params:{NonTrainable_params}")
    
    #*========================================================

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(single_model.cuda(args.local_rank))
    model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[args.local_rank],
                        find_unused_parameters=True,broadcast_buffers = True) 
    # model = torch.nn.parallel.DistributedDataParallel(model) #* 还是会出错, default setting 不行
    # model = torch.nn.DataParallel(model,device_ids=[args.local_rank])
    
    logger.info("construct model done ")
    # logger.info(single_model)

    cudnn.benchmark = args.cudnn_benchmark
    #*=====================================
    atten_criterion = AttentionLoss2().cuda(args.local_rank)
    focal_criterion = SegmentationLosses(weight=None, cuda=True).build_loss(mode='focal')

    inverse_form_criterion = InverseTransform2D()



    #* 不能整除怎么办
    if (args.batch_size % args.nprocs) != 0 and args.local_rank==0 :
        args.batch_size  = int(args.batch_size / args.nprocs)  + args.batch_size % args.nprocs #* 不能整除的部分加到第一个GPU
    else :
        args.batch_size = int(args.batch_size / args.nprocs)

    #* Data loading code
    logger.info(f"rank = {args.local_rank},batch_size == {args.batch_size}")

    train_dataset = Mydataset(root_path=args.train_dir, split='trainval', crop_size=args.crop_size)
    train_sampler = DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,batch_size=args.batch_size, num_workers=args.workers,
            pin_memory=True, drop_last=True, sampler=train_sampler
    )

    logger.info("Dataloader  init  done ")


    #* load test data =====================
    test_loader= None
    if args.validation:
        test_dataset = Mydataset(root_path=args.test_dir, split='test', crop_size=args.crop_size)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, 
                                shuffle=False,num_workers=args.workers,pin_memory=False)
    #*=====================================
    # define loss function (criterion) and pptimizer
    optimizer = torch.optim.SGD(model.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    
    start_epoch = 0
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume,map_location=torch.device(args.local_rank))
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
    
    best_ods = best_ois= best_ap = 0
    for epoch in range(start_epoch, args.epochs):
        lr = adjust_learning_rate(args, optimizer, epoch)
        logger.info('Epoch: [{0}]\tlr {1:.06f}'.format(epoch, lr))
        train_sampler.set_epoch(epoch)        

        #!=============== 检查model的一致性
        # if epoch != 0  and epoch%10 == 0 :
        #     check_checkpoint_path = osp.join(model_save_dir,'ckpt_rank%03d_ep%04d.pth.tar'%(args.local_rank,epoch-1))
        #     is_consistent,_,_=is_model_consistent(
        #         check_checkpoint_path.replace("ckpt_rank%03d"%(args.local_rank),"ckpt_rank%03d"%(1)),
        #         check_checkpoint_path.replace("ckpt_rank%03d"%(args.local_rank),"ckpt_rank%03d"%(0))
        #         )
        #     logger.info(f"is consistent: {is_consistent}")
        #     wandb.log({"is_consistent":1 if is_consistent else 0 })
        #!===============
        train_cerberus(train_loader, model, atten_criterion,
             focal_criterion,optimizer, epoch,_moo = args.moo,
             local_rank = args.local_rank,print_freq=1,
             bg_weight=args.bg_weight,rind_weight=args.rind_weight,
             extra_loss_weight = args.extra_loss_weight,
             inverse_form_criterion=inverse_form_criterion
             )


        #* save model every 20 epoch
        #* 要么 要验证  那就在250epoch之后每5个epoch 验证一次 ,  
        if (args.validation  and (epoch%5==0 or  epoch+1 == args.epochs )and epoch >=260  and  args.local_rank == 0 ): 
            val_dir = osp.join(model_save_dir,'..','ckpt_ep%04d'%epoch)
            os.makedirs(val_dir)
            val_res = edge_validation(model,test_loader,val_dir)
            wandb.log(val_res["Average"])
            logger.info(val_res["Average"])
            save_flag = False 
            if best_ods <val_res["Average"]["ODS"]:
                best_ods=  val_res["Average"]["ODS"]
                save_flag  = True
                logger.info(f" ODS achieve best : {val_res['Average']['ODS']}")
                
            if best_ois <val_res["Average"]["OIS"]:
                best_ois= val_res["Average"]["OIS"]
                save_flag  = True
                logger.info(f" OIS achieve best : {val_res['Average']['OIS']}")
                
                
            if best_ap < val_res["Average"]["AP"]:
                best_ap =  val_res["Average"]["AP"]
                save_flag  = True
                logger.info(f" AP achieve best : {val_res['Average']['AP']}")

            checkpoint_path = osp.join(model_save_dir,\
                'ckpt_rank%03d_ep%04d.pth.tar'%(args.local_rank,epoch))
            
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': val_res["Average"]["AP"],
            }, save_flag, filename=checkpoint_path)
        #* 要么就每30epoch 保存一次 
        # elif((epoch%30==0 or  epoch+1 == args.epochs )and args.local_rank == 0):
        elif(args.local_rank == 0):#* 每个epoch都保存
            checkpoint_path = osp.join(model_save_dir,'ckpt_rank%03d_ep%04d.pth.tar'%(args.local_rank,epoch))
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': None,
            }, True, filename=checkpoint_path)
        else:
            pass
        
    logger.info("train finish!!!! ")

    

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
    if args.save_dir is None :
        args.save_dir = osp.join(osp.dirname(osp.abspath(__file__)),"networks",\
                "lr@%s_ep@%s_bgw@%s_rindw@%s_%s"%(args.lr,args.epochs,args.bg_weight,args.rind_weight,int(time.time())),\
                "checkpoints")

    logger.info(args.save_dir)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    
    # port = int(1e4+np.random.randint(1,10000,1)[0])
    # logger.info(f"port == {port}")
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = str(port)
    # logger.info(f"node number == {args.nprocs}")
    args.nprocs = torch.cuda.device_count()#* gpu  number 

    torch.autograd.set_detect_anomaly(True) 

    train_seg_cerberus(args)
    


def setup_seed(seed):
     torch.manual_seed(seed)
     #*===================
     torch.cuda.manual_seed_all(seed)
     #*===================
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

    
if __name__ == '__main__':
    # 设置随机数种子
    setup_seed(20)
    main()

    