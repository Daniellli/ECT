###
 # @Author: xushaocong
 # @Date: 2022-05-12 21:59:18
 # @LastEditTime: 2022-05-13 18:15:56
 # @LastEditors: xushaocong
 # @Description: 
 # @FilePath: /Cerberus-main/my_script/test.sh
 # email: xushaocong@stu.xmu.edu.cn
### 


# CUDA_VISIBLE_DEVICES=0 python main.py test -d [dataset_path]  \
# -s 512 --resume model_best.pth.tar --phase val --batch-size 1 --ms --workers 10


python main.py test  -s 1024 --resume ./networks/model_best.pth.tar \
--phase val --batch-size 1 --ms --workers 20 --classes 1 --arch test_arch \
2>&1 | tee -a logs/test.log

