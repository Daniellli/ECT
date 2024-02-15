'''
Author: daniel
Date: 2023-02-25 20:21:58
LastEditTime: 2024-02-15 21:31:02
LastEditors: daniel
Description: 
FilePath: /Cerberus-main/utils/edge_option.py
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
    parser.add_argument('cmd', choices=['train', 'test'])
    
    #* unknown 
    parser.add_argument('--step', type=int, default=200)


    #* dataset specific 
    parser.add_argument('-s', '--crop-size', default=320, type=int)

    
    #* train
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('-j', '--workers', type=int, default=16)
    parser.add_argument('--gpu-ids', default='7', type=str)

    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train (default: 300)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--lr-mode', type=str, default='poly')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument("--local_rank", type=int,default=-1,help="node rank for distrubuted training")
    parser.add_argument('--cudnn_benchmark', type=bool, default=True,
                        help='Set cudnn benchmark on (1) or off (0) (default is on).')
                        
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    

    #* hyper parameter 
    parser.add_argument('--arch',type=str, default="test_arch",help='save_name dir ')#* useless
    parser.add_argument("--bg-weight", type=float,default=1,help=" background weight  ")
    parser.add_argument("--rind-weight", type=float,default=1,help=" rind weight  ")
    parser.add_argument("--extra-loss-weight", type=float,default=0.1,help="for evaluation ")
    parser.add_argument("--edge-loss-gamma", type=float,default=0.5,help="for loss ")
    parser.add_argument("--edge-loss-beta", type=float,default=4,help="for loss ")
    parser.add_argument("--rind-loss-gamma", type=float,default=0.5,help="for loss ")
    parser.add_argument("--rind-loss-beta", type=float,default=4,help="for loss ")


    parser.add_argument('--cause-token-num', type=int, default=4, help=' cause token number ')

    #* save path 
    parser.add_argument("--bsds-dir",type=str,default="data/BSDS-RIND/BSDS-RIND/Augmentation/", help="训练数据集的文件夹root")
    parser.add_argument('--wandb', action='store_true',help=' using wandb for log')
    parser.add_argument("--print-freq", type=int,default=10)
    parser.add_argument("--val-freq", type=int,default=5)
    parser.add_argument("--save-freq", type=int,default=5)
    parser.add_argument('--eval_dataset',type=str, default="SBU",help='the eval dataset')
                
    args = parser.parse_args()
    # if args.bn_sync:
    #     drn.BatchNorm = batchnormsync.BatchNormSync

    return args


