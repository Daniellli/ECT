###
 # @Author: xushaocong
 # @Date: 2022-05-12 21:59:29
 # @LastEditTime: 2022-06-13 16:46:44
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



#* 成功在moo == False ,成功训练5个epoch
#python  -m torch.distributed.launch --nproc_per_node=6  --master_port 29504 main3.py \
#train  -s 320 --batch-size 4  --epochs 100 --lr 1e-4 --momentum 0.9 \
#--lr-mode poly --workers 12 --distributed_train --gpu-ids '0,3,4,5,6,7' \
#2>&1 | tee -a logs/train.log


#* moo == True , 进一步封装 DPT , 存在log无法正常输出的问题, 但是可以正常train
# python  -m torch.distributed.launch --nproc_per_node=4  --master_port 29504 main4.py \
# train  -s 320 --batch-size 4  --epochs 300 --lr 1e-4 --momentum 0.9 \
# --lr-mode poly --workers 12 --distributed_train --gpu-ids '3,4,5,7' --moo \
# 2>&1 | tee -a logs/train.log

#* moo == True , 
python  -m torch.distributed.launch --nproc_per_node=2  --master_port 29505 main3.py \
train  -s 320 --batch-size 4  --epochs 300 --lr 1e-4 --momentum 0.9 \
--lr-mode poly --workers 12 --distributed_train --gpu-ids '0,1' \
2>&1 | tee -a logs/train.log



#* resume from last model  
# python  -m torch.distributed.launch --nproc_per_node=2  --master_port 29505 main3.py \
# train  -s 320 --batch-size 4  --epochs 300 --lr 1e-4 --momentum 0.9 \
# --lr-mode poly --workers 12 --distributed_train --gpu-ids '4,5' \
# --resume "/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/networks/dashing-wind-713/checkpoints/checkpoint_ep0099.pth.tar" \
# 2>&1 | tee -a logs/train.log



#* single gpu mode 
# CUDA_LAUNCH_BLOCKING=1 
# python   -u  main3.py \
# train   -s 512 --batch-size 2  --epochs 5 \
#  --lr 1e-5 --momentum 0.9 \
# --lr-mode poly --workers 12 \
# 2>&1 | tee -a logs/train.log
