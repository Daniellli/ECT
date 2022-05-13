###
 # @Author: xushaocong
 # @Date: 2022-05-12 21:59:18
 # @LastEditTime: 2022-05-13 11:05:17
 # @LastEditors: xushaocong
 # @Description: 
 # @FilePath: /Cerberus-main/my_script/test.sh
 # email: xushaocong@stu.xmu.edu.cn
### 


# CUDA_VISIBLE_DEVICES=0 python main.py test -d [dataset_path]  \
# -s 512 --resume model_best.pth.tar --phase val --batch-size 1 --ms --workers 10


CUDA_VISIBLE_DEVICES=1
python main.py test  -s 512 --resume ./networks/model_best.pth.tar \
--phase val --batch-size 1 --ms --workers 10 --classes 1 \
2>&1 | tee -a logs/test.log

