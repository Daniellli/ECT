
###
 # @Author: daniel
 # @Date: 2023-02-06 20:17:43
 # @LastEditTime: 2023-03-17 19:54:16
 # @LastEditors: daniel
 # @Description: 
 # @FilePath: /cerberus/scripts/train_sbd.sh
 # have a nice day
### 




gpuids="0,2,3,5,6,7";
gpu_number=5;

lr=3e-3;
batch_size=2;
epoch=100;
data_dir=data/sbd-preprocess/data_proc;
dataset='sbd';
data_size=352;


scheduler='poly';
decay_rate=0.9; #* power 

# scheduler='step';
# decay_epoch="100 200"
# decay_rate=0.1; #* power 
# --lr-decay-epochs $decay_epoch 

print_freq=10;
val_freq=3;
save_freq=3;

# model2resume=/home/DISCOVER_summer2022/xusc/exp/cerberus/networks/2023-03-02-15:34:1677742465/checkpoints/ckpt_rank000_ep0020.pth.tar;
# --resume $model2resume 
# --weight-decay 1e-4
# --gpu-ids $gpuids 


CUDA_VISIBLE_DEVICES=$gpuids python  -m torch.distributed.launch --nproc_per_node=$gpu_number --master_port 29510 \
se_trainer.py train  -s $data_size --batch-size $batch_size  --epochs $epoch --lr $lr \
--data-dir $data_dir --lr-scheduler $scheduler --lr-decay-rate $decay_rate  --val-all-in 5 \
--dataset $dataset  --val-freq $val_freq --save-freq $save_freq --print-freq $print_freq --wandb \
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
