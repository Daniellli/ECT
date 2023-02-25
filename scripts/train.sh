
###
 # @Author: daniel
 # @Date: 2023-02-06 20:17:43
 # @LastEditTime: 2023-02-25 20:30:45
 # @LastEditors: daniel
 # @Description: 
 # @FilePath: /Cerberus-main/scripts/train.sh
 # have a nice day
### 




gpuids="1";
gpu_number=1;


lr=1e-5;
batch_size=4;
epoch=300;
bg_weights=0.5;
rind_weights=1;
extra_loss_weight=1000;
edge_loss_beta=1;
edge_loss_gamma=0.3;
rind_loss_beta=5;
rind_loss_gamma=0.3


python  -m torch.distributed.launch --nproc_per_node=$gpu_number   --master_port 29510 \
train.py train  -s 320 --batch-size $batch_size  --epochs $epoch --lr $lr --momentum 0.9 \
--lr-mode poly --workers 12 --gpu-ids $gpuids --bg-weight $bg_weights --rind-weight $rind_weights \
--extra-loss-weight $extra_loss_weight --edge-loss-gamma $edge_loss_gamma --edge-loss-beta $edge_loss_beta \
--rind-loss-gamma $rind_loss_gamma  --rind-loss-beta $rind_loss_beta --constraint-loss \
2>&1 | tee -a logs/train.log

# --wandb
