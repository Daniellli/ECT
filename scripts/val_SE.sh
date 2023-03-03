
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
data_dir=data/cityscapes-preprocess/data_proc;
dataset='cityscapes';
data_size=640;


#* validate all model 
gpu_num=1;
port=29550;

# val_dir='/DATA2/xusc/cerberus/networks/2023-03-02-15:07:1677740825/checkpoints/'

# python -m torch.distributed.launch --nproc_per_node=$gpu_num --master_port $port \
# train_SE.py val  -s $data_size --batch-size $bs --gpu-ids $gpuids --workers 8 \
# --data-dir $data_dir --dataset $dataset --resume-model-dir $val_dir \
# 2>&1 | tee -a logs/validate.log



# python -c "import torch; print(torch.cuda.is_available(),torch.version.cuda);"



# model2resume=/DATA2/xusc/cerberus/networks/2023-03-02-15:07:1677740825/checkpoints/ckpt_rank000_ep0106.pth.tar;


model2resume=/home/DISCOVER_summer2022/xusc/exp/cerberus/networks/2023-03-02-15:34:1677742465/checkpoints/ckpt_rank000_ep0020.pth.tar;

#* for sbd 
data_dir=data/sbd-preprocess/data_proc;
dataset='sbd';
data_size=512;
batch_size=32;


# #* test 
python -m torch.distributed.launch --nproc_per_node=$gpu_num --master_port $port \
train_SE.py test  -s $data_size --batch-size $batch_size --gpu-ids $gpuids --workers 8 \
--data-dir $data_dir --dataset $dataset --resume $model2resume \
2>&1 | tee -a logs/test_cityscapes.log
