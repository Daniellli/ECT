###
 # @Author: xushaocong
 # @Date: 2022-05-12 21:59:29
 # @LastEditTime: 2022-06-08 12:00:03
 # @LastEditors: xushaocong
 # @Description: 
 # @FilePath: /Cerberus-main/my_script/train.sh
 # email: xushaocong@stu.xmu.edu.cn
### 


# python main.py train -d ./dataset/nyud2 \
# -s 512 --batch-size 1 --random-scale 2 --random-rotate 10 \
# --epochs 200 --lr 0.007 --momentum 0.9 --lr-mode poly --workers 12 \
# --classes 1 2>&1 | tee -a logs/train.log
# CUDA_VISIBLE_DEVICES=0,1,2

# python   -W ignore -m torch.distributed.launch --nproc_per_node=3 main.py \
# train -d ./dataset/nyud2 -s 512 --batch-size 1 --random-scale 2 \
# --random-rotate 10 --epochs 5 --lr 0.007 --momentum 0.9 \
# --lr-mode poly --workers 12 --classes 1 \
# 2>&1 | tee -a logs/train.log

# python  -m torch.distributed.launch --nproc_per_node=4  --master_port 29504 main3.py \
# train  -s 512 --batch-size 4 --random-scale 2 \
# --random-rotate 10 --epochs 5 --lr 1e-5 --momentum 0.9 \
# --lr-mode poly --workers 12 --classes 2 --distributed_train --gpu-ids '4,5,6,7' \
# 2>&1 | tee -a logs/train.log

#* 成功在moo == False ,训练5个epoch
# python  -m torch.distributed.launch --nproc_per_node=4  --master_port 29504 main3.py \
# train  -s 512 --batch-size 4  --epochs 5 --lr 1e-4 --momentum 0.9 \
# --lr-mode poly --workers 12 --distributed_train --gpu-ids '4,5,6,7' \
# 2>&1 | tee -a logs/train.log

#* 成功在moo == True ,训练5个epoch
python  -m torch.distributed.launch --nproc_per_node=4  --master_port 29504 main3.py \
train  -s 512 --batch-size 4  --epochs 300 --lr 1e-4 --momentum 0.9 \
--lr-mode poly --workers 12 --distributed_train --gpu-ids '4,5,6,7' \
2>&1 | tee -a logs/train.log

#* 原来的训练代码
# python   -u  main.py \
# train   -s 512 --batch-size 1 --random-scale 2 \
# --random-rotate 10 --epochs 5 --lr 0.007 --momentum 0.9 \
# --lr-mode poly --workers 12 --classes 1 \
# 2>&1 | tee -a logs/train.log

#* 取消数据增强
#* -s 是crop_size , 就是输出模型的数据大小
# CUDA_LAUNCH_BLOCKING=1 
# python   -u  main3.py \
# train   -s 512 --batch-size 2  --epochs 5 \
# --lr 1e-5 --momentum 0.9 \
# --lr-mode poly --workers 12 \
# 2>&1 | tee -a logs/train.log