###
 # @Author: xushaocong
 # @Date: 2022-05-12 21:59:29
 # @LastEditTime: 2022-05-13 14:03:58
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

python   -W ignore -m torch.distributed.launch --nproc_per_node=3 main.py \
train -d ./dataset/nyud2 -s 512 --batch-size 1 --random-scale 2 \
--random-rotate 10 --epochs 5 --lr 0.007 --momentum 0.9 \
--lr-mode poly --workers 12 --classes 1 \
2>&1 | tee -a logs/train.log
