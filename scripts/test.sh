###
 # @Author: xushaocong
 # @Date: 2022-05-12 21:59:18
 # @LastEditTime: 2023-03-15 14:51:16
 # @LastEditors: daniel
 # @Description: 
 # @FilePath: /cerberus/scripts/test.sh
 # email: xushaocong@stu.xmu.edu.cn
### 


# CUDA_VISIBLE_DEVICES=0 python main.py test -d [dataset_path]  \
# -s 512 --resume model_best.pth.tar --phase val --batch-size 1 --ms --workers 10

#* 在训练集上测试精度

# python main.py test  -s 1024 --resume ./model_best.pth.tar \
# --phase train  --batch-size 1 --ms --workers 20 --classes 1 --arch test_arch \
# --with-gt 2>&1 | tee -a logs/test.log


#* 在测试集上测试精度
# python main.py test  -s 1024 --resume ./model_best.pth.tar \
# --phase test  --batch-size 1 --ms --workers 20 --classes 1 --arch test_arch \
# --with-gt 2>&1 | tee -a logs/test.log

#* 在验证集上测试精度
# python main.py test  -s 1024 --resume ./model_best.pth.tar \
# --phase val  --batch-size 1 --ms --workers 20 --classes 1 --arch test_arch \
# --with-gt 2>&1 | tee -a logs/test.log


#* rindnet  在测试集上测试精度, model绝对路径, 只能绝对路径

# for lr in $(seq 3 1 5 ); do 
#     echo $lr;
#     python -u main4.py test  -s 320 \
#     --resume /home/DISCOVER_summer2022/xusc/exp/Cerberus-main/networks/le-$lr/checkpoints/edge_cerberus_le-$lr.pth.tar \
#     --batch-size 1 --workers 40 \
#     2>&1 | tee -a logs/test.log
# done;





#* test one model 


# resume_model="/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/networks/need2release/checkpoints/full_version.pth.tar";
# save_dir='nyud2'


# python -u test_NYU.py test  -s 320 \
# --resume $resume_model \
# --batch-size 1 --workers 40 --gpu-ids "2" --run-id 2 --save-file $save_dir \
# 2>&1 | tee -a logs/test_iiw.log





resume_model="/data3/xusc/exp/cerberus/networks/2023-03-14-04:55:1678740937/checkpoints/ckpt_rank000_ep0299.pth.tar";
save_dir='val_new_model'


python -u test.py test  -s 320 \
--resume $resume_model \
--batch-size 1 --workers 40 --gpu-ids "1" --run-id 2 --save-file $save_dir \
2>&1 | tee -a logs/test.log







#* test all model under path 
# path=/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/networks/final_version/edge_final_8_3090/

# for model in $(ls $path); do 
#     echo $path$model;
#     model_name=(${model//./ });
#     echo ${model_name[0]}${model_name[1]};
#     python -u test.py test  -s 320 \
#     --resume $path$model \
#     --batch-size 1 --workers 40 --run-id 0 --save-file ${model_name[0]}${model_name[1]} \
#     2>&1 | tee -a logs/test.log

# done;
# echo $idx;