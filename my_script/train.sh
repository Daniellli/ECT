###
 # @Author: xushaocong
 # @Date: 2022-05-12 21:59:29
 # @LastEditTime: 2022-08-09 23:20:44
 # @LastEditors: xushaocong
 # @Description: 
 # @FilePath: /cerberus/my_script/train.sh
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
batch_size=12;
gpuids="0,1,3";
gpu_number=3;


# batch_size=8;
# gpuids="6,7";
# gpu_number=2;
epoch=300;
bg_weights=(0.5);
rind_weights=(1);
extra_loss_weight=(1000);#* 这个权重 影响不大,  不管是 (0.1 1 0.01 10 1000 )都差不多

#* 搜索策略, 固定gamma 搜 beta , 固定beta搜gamma
#* beta = [1 to 5 ]
#* gamma = [0.1 to 0.8 ]
#* by default : beta = 4, gamma=0.5
edge_loss_beta=(4);
edge_loss_gamma=(0.1 0.2 0.3 0.4 0.6 0.7 0.8);
rind_loss_beta=(4);
rind_loss_gamma=(0.5);

for idx in $(seq 0 1 6);do 
    
    #*========================================================================================
    # echo edge_loss_beta@${edge_loss_beta[$idx]};
    echo edge_loss_gamma@${edge_loss_gamma[$idx]};
    # echo rind_loss_beta@${rind_loss_beta[$idx]};
    # echo rind_loss_gamma@${rind_loss_gamma[$idx]};

    python  -m torch.distributed.launch --nproc_per_node=$gpu_number   --master_port 29510 \
        train2.py train  -s 320 --batch-size $batch_size  --epochs $epoch --lr $lr --momentum 0.9 \
        --lr-mode poly --workers 12 --gpu-ids $gpuids --bg-weight ${bg_weights[0]} --rind-weight ${rind_weights[0]} \
        --extra-loss-weight ${extra_loss_weight[0]} --edge-loss-gamma ${edge_loss_gamma[$idx]} --edge-loss-beta ${edge_loss_beta[0]} \
        --rind-loss-gamma 0.5  --rind-loss-beta 4 --wandb \
        2>&1 | tee -a logs/train.log
    #*=======================================================git a=================================


    #*========================================================================================resume 
    #* --resume 绝对路径和 相对路径都可以 
    # python  -m torch.distributed.launch --nproc_per_node=$gpu_number   --master_port 29506 \
    #     train2.py train  -s 320 --batch-size $batch_size  --epochs $epoch --lr $lr --momentum 0.9 \
    #     --resume "/home/DISCOVER_summer2022/xusc/exp/cerberus/networks/lr@1e-05_ep@300_bgw@2.0_rindw@1.0_1659166908/checkpoints/ckpt_rank000_ep0020.pth.tar" \
    #     --lr-mode poly --workers 12 --gpu-ids $gpuids --bg-weight ${bg_weights[0]} --rind-weight ${rind_weights[0]} \
    #     --save-dir "/home/DISCOVER_summer2022/xusc/exp/cerberus/networks/lr@1e-05_ep@300_bgw@2.0_rindw@1.0_1659166908/checkpoints/" \
    #     2>&1 | tee -a logs/train.log
    #*========================================================================================

done


