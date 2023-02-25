





import argparse


'''
description:  parse args
return {*}
'''
def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('cmd', choices=['train', 'test'])
    parser.add_argument('-d', '--data-dir', default='./dataset/BSDS_RIND_mine')
    parser.add_argument('-s', '--crop-size', default=320, type=int)
    parser.add_argument('--step', type=int, default=200)
    parser.add_argument('--arch',type=str, default="test_arch",help='save_name dir ')
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--lr-mode', type=str, default='poly')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    #todo  : eval during train 
    parser.add_argument('-e', '--evaluate', dest='evaluate',
                        action='store_true',help='evaluate model on validation set')

    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    parser.add_argument('--pretrained-model', dest='pretrained_model',
                        default='', type=str, metavar='PATH',
                        help='use pre-trained model')

    parser.add_argument('-j', '--workers', type=int, default=10)
    parser.add_argument('--bn-sync', action='store_true')#* 暂时没用
    parser.add_argument('--gpu-ids', default='7', type=str)
    parser.add_argument('--moo', action='store_true',
                        help='Turn on multi-objective optimization')
    parser.add_argument("--local_rank", type=int,default=-1,help="node rank for distrubuted training")
    parser.add_argument('--cudnn_benchmark', type=bool, default=True,
                        help='Set cudnn benchmark on (1) or off (0) (default is on).')

    parser.add_argument("--run-id", type=int,default=None,help="for evaluation ")

    parser.add_argument("--extra-loss-weight", type=float,default=0.1,help="for evaluation ")


    parser.add_argument('--validation', action='store_true',help='validate model during training ')


    parser.add_argument("--bg-weight", type=float,default=1,help=" background weight  ")
    parser.add_argument("--rind-weight", type=float,default=1,help=" rind weight  ")

    parser.add_argument("--train-dir",type=str,default="data/BSDS-RIND/BSDS-RIND/Augmentation/",
                help="训练数据集的文件夹root")

    parser.add_argument("--test-dir",type=str,default="data/BSDS-RIND/BSDS-RIND/Augmentation/",
                help="训练数据集的文件夹root")

    parser.add_argument("--save-dir",type=str,default=None,
                help="save path")
    
    parser.add_argument('--wandb', action='store_true',help=' using wandb for log')

    parser.add_argument('--constraint-loss', action='store_true',help='using constraint loss or not')
    
    
    parser.add_argument("--save-file",type=str,default=None,
                help="save path name")

    parser.add_argument("--edge-loss-gamma", type=float,default=0.5,help="for loss ")
    parser.add_argument("--edge-loss-beta", type=float,default=4,help="for loss ")
    parser.add_argument("--rind-loss-gamma", type=float,default=0.5,help="for loss ")
    parser.add_argument("--rind-loss-beta", type=float,default=4,help="for loss ")

                
    args = parser.parse_args()
    # if args.bn_sync:
    #     drn.BatchNorm = batchnormsync.BatchNormSync

    return args


