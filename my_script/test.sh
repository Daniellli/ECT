###
 # @Author: xushaocong
 # @Date: 2022-05-12 21:59:18
 # @LastEditTime: 2022-05-16 09:18:44
 # @LastEditors: xushaocong
 # @Description: 
 # @FilePath: /cerberus/my_script/test.sh
 # email: xushaocong@stu.xmu.edu.cn
### 


# CUDA_VISIBLE_DEVICES=0 python main.py test -d [dataset_path]  \
# -s 512 --resume model_best.pth.tar --phase val --batch-size 1 --ms --workers 10

#* 在训练集上测试精度

# python main.py test  -s 1024 --resume ./model_best.pth.tar \
# --phase train  --batch-size 1 --ms --workers 20 --classes 1 --arch test_arch \
# --with-gt 2>&1 | tee -a logs/test.log


#* 在测试集上测试精度
python main.py test  -s 1024 --resume ./model_best.pth.tar \
--phase test  --batch-size 1 --ms --workers 20 --classes 1 --arch test_arch \
--with-gt 2>&1 | tee -a logs/test.log

#* 在验证集上测试精度
# python main.py test  -s 1024 --resume ./model_best.pth.tar \
# --phase val  --batch-size 1 --ms --workers 20 --classes 1 --arch test_arch \
# --with-gt 2>&1 | tee -a logs/test.log


