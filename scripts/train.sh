
###
 # @Author: daniel
 # @Date: 2023-02-06 20:17:43
 # @LastEditTime: 2023-03-16 00:37:48
 # @LastEditors: daniel
 # @Description: 
 # @FilePath: /cerberus/scripts/train.sh
 # have a nice day
### 




gpuids="1,2,3,5,7";
gpu_number=5;

lr=1e-5;
batch_size=2;
epoch=320;
bg_weights=0.5;
rind_weights=1;
extra_loss_weight=1000;
edge_loss_beta=1;
edge_loss_gamma=0.3;
rind_loss_beta=5;
rind_loss_gamma=0.3

# python  -m torch.distributed.launch --nproc_per_node=$gpu_number   --master_port 29510 \
# train.py train  -s 320 --batch-size $batch_size  --epochs $epoch --lr $lr --momentum 0.9 \
# --lr-mode poly --workers 12 --gpu-ids $gpuids --bg-weight $bg_weights --rind-weight $rind_weights \
# --extra-loss-weight $extra_loss_weight --edge-loss-gamma $edge_loss_gamma --edge-loss-beta $edge_loss_beta \
# --rind-loss-gamma $rind_loss_gamma  --rind-loss-beta $rind_loss_beta --constraint-loss --wandb \
# 2>&1 | tee -a logs/train.log


#* trainer version
# resume=/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/networks/need2release/checkpoints/full_version.pth.tar;

CUDA_VISIBLE_DEVICES=$gpuids python  -m torch.distributed.launch --nproc_per_node=$gpu_number   --master_port 29510 \
trainer.py train  -s 320 --batch-size $batch_size  --epochs $epoch --lr $lr --momentum 0.9 \
--lr-mode poly --workers 16 --gpu-ids $gpuids --bg-weight $bg_weights --rind-weight $rind_weights \
--extra-loss-weight $extra_loss_weight --edge-loss-gamma $edge_loss_gamma --edge-loss-beta $edge_loss_beta \
--rind-loss-gamma $rind_loss_gamma  --rind-loss-beta $rind_loss_beta  \
2>&1 | tee -a logs/train.log

# --wandb
#  --resume $resume

