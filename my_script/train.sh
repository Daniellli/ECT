###
 # @Author: xushaocong
 # @Date: 2022-05-12 21:59:29
 # @LastEditTime: 2022-08-18 16:14:16
 # @LastEditors: xushaocong
 # @Description: 
 # @FilePath: /Cerberus-main/my_script/train.sh
 # email: xushaocong@stu.xmu.edu.cn
### 

#* 原来的训练代码
# python   -u  main.py \
# train   -s 512 --batch-size 1 --random-scale 2 \
# --random-rotate 10 --epochs 5 --lr 0.007 --momentum 0.9 \
# --lr-mode poly --workers 12 --classes 1 \
# 2>&1 | tee -a logs/train.log



#* 进一步封装 DPT
# python   main4.py train  -s 320 --batch-size 12  --epochs 1 --lr 1e-4 --momentum 0.9 \
# --lr-mode poly --workers 12 --gpu-ids '2,4,5' \
# 2>&1 | tee -a logs/train.log



#* 炼丹代码
lr=1e-5;

# batch_size=32;
# gpuids="0,1,2,3,4,5,6,7";
# gpu_number=8;

batch_size=4;
gpuids="0,1";
gpu_number=2;

epoch=300;
bg_weights=(0.5);
rind_weights=(1);
extra_loss_weight=(1000);#* 这个权重 影响不大,  不管是 (0.1 1 0.01 10 1000 )都差不多

#* 搜索策略, 固定gamma 搜 beta , 固定beta搜gamma
#* beta = [1 to 5 ]
#* gamma = [0.1 to 0.8 ]
#* by default : beta = 4, gamma=0.5
edge_loss_beta=(1);
edge_loss_gamma=(0.3);
rind_loss_beta=(5);#* 好像越高越好
rind_loss_gamma=(0.3);

for idx in $(seq 0 1 0);do 
    
    #*========================================================================================
    # echo edge_loss_beta@${edge_loss_beta[$idx]};
    # echo edge_loss_gamma@${edge_loss_gamma[$idx]};
    # echo rind_loss_beta@${rind_loss_beta[$idx]};
    # echo rind_loss_gamma@${rind_loss_gamma[$idx]};

    python  -m torch.distributed.launch --nproc_per_node=$gpu_number   --master_port 29510 \
        train2.py train  -s 320 --batch-size $batch_size  --epochs $epoch --lr $lr --momentum 0.9 \
        --lr-mode poly --workers 12 --gpu-ids $gpuids --bg-weight ${bg_weights[0]} --rind-weight ${rind_weights[0]} \
        --extra-loss-weight ${extra_loss_weight[0]} --edge-loss-gamma ${edge_loss_gamma[0]} --edge-loss-beta ${edge_loss_beta[0]} \
        --rind-loss-gamma ${rind_loss_gamma[0]}  --rind-loss-beta ${rind_loss_beta[0]} --wandb --constraint-loss \
        2>&1 | tee -a logs/train.log

    # python  -m torch.distributed.launch --nproc_per_node=$gpu_number   --master_port 29510 \
    #     train2.py train  -s 320 --batch-size $batch_size  --epochs $epoch --lr $lr --momentum 0.9 \
    #     --lr-mode poly --workers 12 --gpu-ids $gpuids --bg-weight ${bg_weights[0]} --rind-weight ${rind_weights[0]} \
    #     --extra-loss-weight ${extra_loss_weight[0]} --edge-loss-gamma ${edge_loss_gamma[0]} --edge-loss-beta ${edge_loss_beta[0]} \
    #     --rind-loss-gamma ${rind_loss_gamma[0]}  --rind-loss-beta ${rind_loss_beta[0]} --wandb \
    #     2>&1 | tee -a logs/train.log
    #*=======================================================git a=================================


    #*========================================================================================resume 
    #* --resume 绝对路径和 相对路径都可以 
    
    python  -m torch.distributed.launch --nproc_per_node=$gpu_number   --master_port 29506 \
        train2.py train  -s 320 --batch-size $batch_size  --epochs $epoch --lr $lr --momentum 0.9 \
        --resume "/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/networks/rind_loss_beta@5.000000_rind_loss_gamma@0.300000_edge_loss_beta@1.000000_edge_loss_gamma0.300000_1660795253/checkpoints/ckpt_rank000_ep0255.pth.tar" \
        --lr-mode poly --workers 12 --gpu-ids $gpuids --bg-weight ${bg_weights[0]} --rind-weight ${rind_weights[0]} \
        --extra-loss-weight ${extra_loss_weight[0]} --edge-loss-gamma ${edge_loss_gamma[0]} --edge-loss-beta ${edge_loss_beta[0]} \
        --rind-loss-gamma ${rind_loss_gamma[0]}  --rind-loss-beta ${rind_loss_beta[0]} --wandb --validation \
        --save-dir "/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/networks/rind_loss_beta@5.000000_rind_loss_gamma@0.300000_edge_loss_beta@1.000000_edge_loss_gamma0.300000_1660795253/checkpoints/" \
        2>&1 | tee -a logs/train.log
    #*========================================================================================

done


