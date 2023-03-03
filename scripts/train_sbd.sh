
###
 # @Author: daniel
 # @Date: 2023-02-06 20:17:43
 # @LastEditTime: 2023-03-01 18:27:05
 # @LastEditors: daniel
 # @Description: 
 # @FilePath: /Cerberus-main/scripts/train_SE.sh
 # have a nice day
### 




gpuids="0,1,2,3";
gpu_number=4;

lr=5e-3;
batch_size=16;
epoch=100;
bg_weights=0.5;
rind_weights=1;
# inverseform_loss_weight=1e+3;
inverseform_loss_weight=1;
edge_loss_beta=1;
edge_loss_gamma=0.3;
rind_loss_beta=5;
rind_loss_gamma=0.3;
data_dir=data/sbd-preprocess/data_proc;
dataset='sbd';
data_size=352;


scheduler='poly';
decay_rate=0.9; #* power 


# scheduler='step';
# decay_epoch="100 200"
# decay_rate=0.1; #* power 
# --lr-decay-epochs $decay_epoch 

print_freq=20;
val_freq=5;
save_freq=5;

# model2resume=/DATA2/xusc/cerberus/networks/2023-03-01-14:37:1677652667/checkpoints/ckpt_rank000_ep0002.pth.tar;


python  -m torch.distributed.launch --nproc_per_node=$gpu_number   --master_port 29510 \
train_SE.py train  -s $data_size --batch-size $batch_size  --epochs $epoch --lr $lr --momentum 0.9 \
--gpu-ids $gpuids --bg-weight $bg_weights --rind-weight $rind_weights --edge-loss-gamma $edge_loss_gamma \
--edge-loss-beta $edge_loss_beta --rind-loss-gamma $rind_loss_gamma  --rind-loss-beta $rind_loss_beta \
--inverseform-loss --inverseform-loss-weight $inverseform_loss_weight --data-dir $data_dir \
--lr-scheduler $scheduler --lr-decay-rate $decay_rate --weight-decay 1e-4 --wandb \
--dataset $dataset  --val-freq $val_freq --save-freq $save_freq --print-freq $print_freq \
2>&1 | tee -a logs/train_sbd.log





# python  -m torch.distributed.launch --nproc_per_node=1   --master_port 29511 \
# train_SE.py train  -s 320 --batch-size 4  --epochs 300 --lr 1e-5 \
# --gpu-ids '1' --bg-weight 0.5 --rind-weight 1 \
# --extra-loss-weight 1e+3 --edge-loss-gamma 0.3 --edge-loss-beta 1 \
# --rind-loss-gamma 0.3  --rind-loss-beta 5 --constraint-loss \
# 2>&1 | tee -a logs/train.log



# resume_model=/home/DISCOVER_summer2022/xusc/exp/cerberus/networks/2023-02-26-13:27:1677389259/checkpoints/ckpt_rank000_ep0015.pth.tar;

#* test 
# gpuids=3;
# python -m torch.distributed.launch --nproc_per_node=1 --master_port 29510 \
# train_SE.py test  -s $data_size --batch-size 2 --gpu-ids $gpuids --workers 1 \
# --data-dir $data_dir --dataset $dataset   --resume $resume_model \
# 2>&1 | tee -a logs/test.log
