'''
Author: daniel
Date: 2023-02-25 20:23:24
LastEditTime: 2023-03-17 13:05:30
LastEditors: daniel
Description:  for semantic edge 
FilePath: /cerberus/utils/semantic_edge_option.py
have a nice day
'''






import argparse


'''
description:  parse args
return {*}
'''
def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='')
    
    parser.add_argument('cmd', choices=['train', 'test','val'])
    #* for training procedure
    parser.add_argument('--step', type=int, default=200)
    parser.add_argument('--arch',type=str, default="test_arch",help='save_name dir ')
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--lr-mode', type=str, default='poly')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
                        
    parser.add_argument('--lr-scheduler', type=str, default='step',
                        choices=["step", "cosine",'step2','poly'])
    parser.add_argument('--lr-decay-epochs', type=int, default=[280, 340],
                        nargs='+', help='when to decay lr, can be a list')
    
    parser.add_argument('--lr-decay-rate', type=float, default=0.1,
                        help='for step scheduler. decay rate for lr')
    
    parser.add_argument('--change-decay-epoch', action='store_true',help=' for resume, change the milestone of loaded schedule ')    


    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    
    parser.add_argument('--resume-model-dir', default=None, type=str, metavar='PATH',
                        help='path to checkpoint directory (default: none)')
    
    parser.add_argument('--pretrained-model', dest='pretrained_model',
                        default='', type=str, metavar='PATH',
                        help='use pre-trained model')
    parser.add_argument('-j', '--workers', type=int, default=12)
    parser.add_argument("--inverseform-loss-weight", type=float,default=1,help="for evaluation ")#* 1e+3 in ECT 
    parser.add_argument("--bg-weight", type=float,default=0.5,help=" background weight  ")
    parser.add_argument("--rind-weight", type=float,default=1,help=" rind weight  ")

    parser.add_argument("--edge-loss-gamma", type=float,default=0.3,help="for loss ")
    parser.add_argument("--edge-loss-beta", type=float,default=1,help="for loss ")
    parser.add_argument("--rind-loss-gamma", type=float,default=0.3,help="for loss ")
    parser.add_argument("--rind-loss-beta", type=float,default=5,help="for loss ")
    
    

    #* for distributed train            
    parser.add_argument('--bn-sync', action='store_true')#* 暂时没用
    # parser.add_argument('--gpu-ids', default='7', type=str)
    parser.add_argument("--local_rank", type=int,default=-1,help="node rank for distrubuted training")
    parser.add_argument('--cudnn_benchmark', type=bool, default=True,
                        help='Set cudnn benchmark on (1) or off (0) (default is on).')



    #* for dataset
    parser.add_argument('--dataset', default='bsds',choices=['cityscapes','sbd','bsds'])
    parser.add_argument('-d', '--data-dir', default='./dataset/BSDS_RIND_mine')
    parser.add_argument('-s', '--crop-size', default=320, type=int)
    parser.add_argument("--train-dir",type=str,default="data/BSDS-RIND/BSDS-RIND/Augmentation/",
                help="训练数据集的文件夹root")
    parser.add_argument("--test-dir",type=str,default="data/BSDS-RIND/BSDS-RIND/Augmentation/",
                help="训练数据集的文件夹root")

    

    #* for log 
    parser.add_argument('--print-freq', default=10, type=int)
    parser.add_argument('--val-freq', default=1, type=int)
    parser.add_argument('--val-all-in', default=60, type=int)
    parser.add_argument('--save-freq', default=1, type=int)
    parser.add_argument("--run-id", type=int,default=None,help="for evaluation ")
    parser.add_argument("--save-dir",type=str,default=None,help="save path")
    parser.add_argument('--wandb', action='store_true',help=' using wandb for log')    

    args = parser.parse_args()
    # if args.bn_sync:
    #     drn.BatchNorm = batchnormsync.BatchNormSync

    return args


