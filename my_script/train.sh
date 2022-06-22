###
 # @Author: xushaocong
 # @Date: 2022-05-12 21:59:29
<<<<<<< HEAD
 # @LastEditTime: 2022-06-22 08:26:20
=======
 # @LastEditTime: 2022-06-21 20:45:22
>>>>>>> 0899f4d6c5da6f1d4761719f3224375f58b7634d
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



#* 成功在moo == False ,成功训练5个epoch
# python  -m torch.distributed.launch --nproc_per_node=2  --master_port 29504 main3.py \
# train  -s 320 --batch-size 4  --epochs 100 --lr 1e-4 --momentum 0.9 \
# --lr-mode poly --workers 12 --distributed_train --gpu-ids '4,5' \
# 2>&1 | tee -a logs/train.log



#* 进一步封装 DPT
# python   main4.py train  -s 320 --batch-size 12  --epochs 1 --lr 1e-4 --momentum 0.9 \
# --lr-mode poly --workers 12 --gpu-ids '2,4,5' \
# 2>&1 | tee -a logs/train.log



#* moo == False , 
lr=1e-5;
<<<<<<< HEAD
batch_size=20;
gpuids="1,3,4,5,6";
epoch=300;
bg_weights=$(seq 0.9 0.02 0.99);
=======
batch_size=128;
gpuids="0,1,2,3";
epoch=300;
bg_weights=$(seq 0.93 0.02 1);
>>>>>>> 0899f4d6c5da6f1d4761719f3224375f58b7634d
# for bg_weight in ${bg_weights[@]};do 
#     echo $bg_weight;
#     python   main4.py train  -s 320 --batch-size $batch_size  --epochs $epoch --lr $lr --momentum 0.9 \
#     --lr-mode poly --workers 12 --gpu-ids $gpuids --bg-weight $bg_weight \
#     2>&1 | tee -a logs/train.log
# done;


#* new arch test 
python   train.py train  -s 320 --batch-size $batch_size  --epochs $epoch --lr $lr --momentum 0.9 \
    --lr-mode poly --workers 12 --gpu-ids $gpuids \
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





