
###
 # @Author: daniel
 # @Date: 2023-02-06 20:17:43
 # @LastEditTime: 2023-03-04 20:28:33
 # @LastEditors: daniel
 # @Description: 
 # @FilePath: /Cerberus-main/scripts/train_SE_cityscapes.sh
 # have a nice day
### 




gpuids="0,1,2,3,4,5";
gpu_number=6;



# lr=8e-2;
#* for CNN
# lr=8e-2;
#* for transformer
lr=1e-5;
batch_size=2;
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
#* for PolynomialLR
scheduler='poly';
decay_rate=0.9; #* power 

#* for MultiStepLR
# scheduler='step';
# decay_epoch="70 110 160";
# decay_rate=0.5;
# decay_epoch="100 200"
#  --lr-decay-epochs $decay_epoch

print_freq=20;
val_freq=5;
save_freq=5;

# model2resume=/DATA2/xusc/cerberus/networks/2023-03-01-16:07:1677658050/checkpoints/model_best.pth.tar;
# --resume $model2resume --change-decay-epoch 


#* train 
python  -m torch.distributed.launch --nproc_per_node=$gpu_number   --master_port 29510 \
train_SE.py train  -s $data_size --batch-size $batch_size  --epochs $epoch --lr $lr --momentum 0.9 \
--gpu-ids $gpuids --bg-weight $bg_weights --rind-weight $rind_weights --edge-loss-gamma $edge_loss_gamma \
--edge-loss-beta $edge_loss_beta --rind-loss-gamma $rind_loss_gamma  --rind-loss-beta $rind_loss_beta \
--inverseform-loss --inverseform-loss-weight $inverseform_loss_weight --data-dir $data_dir --wandb \
--lr-scheduler $scheduler --lr-decay-rate $decay_rate --weight-decay 1e-4 \
--dataset $dataset  --val-freq $val_freq --save-freq $save_freq --print-freq $print_freq \
2>&1 | tee -a logs/train_cityscapes.log



# model2resume=/DATA2/xusc/cerberus/networks/2023-03-02-01:35:1677692146/checkpoints/model_best.pth.tar;
# #* test 
# gpuids=3;
# python -m torch.distributed.launch --nproc_per_node=1 --master_port 29510 \
# train_SE.py test  -s $data_size --batch-size 2 --gpu-ids $gpuids --workers 8 \
# --data-dir $data_dir --dataset $dataset --resume $model2resume \
# 2>&1 | tee -a logs/test_cityscapes.log
