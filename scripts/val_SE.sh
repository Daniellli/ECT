
###
 # @Author: daniel
 # @Date: 2023-02-06 20:17:43
 # @LastEditTime: 2023-02-26 10:24:29
 # @LastEditors: daniel
 # @Description: 
 # @FilePath: /cerberus/scripts/train_SE.sh
 # have a nice day
### 




gpuids="0,1,2,3";
gpu_number=4;


# lr=1e-5;
lr=0.08;
batch_size=4;
epoch=200;
bg_weights=0.5;
rind_weights=1;
# inverseform_loss_weight=1e+3;
inverseform_loss_weight=1;
edge_loss_beta=1;
edge_loss_gamma=0.3;
rind_loss_beta=5;
rind_loss_gamma=0.3;
data_dir=data/cityscapes/data_proc;
dataset='cityscapes';
data_size=640;

# python  -m torch.distributed.launch --nproc_per_node=$gpu_number   --master_port 29510 \
# train_SE.py train  -s $data_size --batch-size $batch_size  --epochs $epoch --lr $lr --momentum 0.9 \
# --gpu-ids $gpuids --bg-weight $bg_weights --rind-weight $rind_weights --edge-loss-gamma $edge_loss_gamma \
# --edge-loss-beta $edge_loss_beta --rind-loss-gamma $rind_loss_gamma  --rind-loss-beta $rind_loss_beta \
# --inverseform-loss --inverseform-loss-weight $inverseform_loss_weight --data-dir $data_dir --wandb \
# --dataset $dataset  --val-freq 1 --save-freq 3 --print-freq 1 2>&1 | tee -a logs/train.log




# python  -m torch.distributed.launch --nproc_per_node=1   --master_port 29511 \
# train_SE.py train  -s 320 --batch-size 4  --epochs 300 --lr 1e-5 \
# --gpu-ids '1' --bg-weight 0.5 --rind-weight 1 \
# --extra-loss-weight 1e+3 --edge-loss-gamma 0.3 --edge-loss-beta 1 \
# --rind-loss-gamma 0.3  --rind-loss-beta 5 --constraint-loss \
# 2>&1 | tee -a logs/train.log








# #* test 
# gpuids=3;
# python -m torch.distributed.launch --nproc_per_node=1 --master_port 29510 \
# train_SE.py test  -s $data_size --batch-size 2 --gpu-ids $gpuids --workers 1 \
# --data-dir $data_dir --dataset $dataset   --resume $resume_model \
# 2>&1 | tee -a logs/test.log

#* validate all model 

gpuids='0,1,2,4,5,7';
gpu_num=6;
port=29550;
bs=4;

python -m torch.distributed.launch --nproc_per_node=$gpu_num --master_port $port \
train_SE.py val  -s $data_size --batch-size $bs --gpu-ids $gpuids --workers 8 \
--data-dir $data_dir --dataset $dataset \
--resume-model-dir '/DATA2/xusc/cerberus/networks/2023-02-28-01:45:1677519956/checkpoints/' \
2>&1 | tee -a logs/validate.log



# python -c "import torch; print(torch.cuda.is_available(),torch.version.cuda);"

