###
 # @Author: xushaocong
 # @Date: 2022-05-12 21:59:29
 # @LastEditTime: 2022-05-12 21:59:30
 # @LastEditors: xushaocong
 # @Description: 
 # @FilePath: /Cerberus-main/my_script/train.sh
 # email: xushaocong@stu.xmu.edu.cn
### 



CUDA_VISIBLE_DEVICES=0 python main.py train -d [dataset_path] \
-s 512 --batch-size 2 --random-scale 2 --random-rotate 10 \
--epochs 200 --lr 0.007 --momentum 0.9 --lr-mode poly --workers 12 