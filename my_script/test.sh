###
 # @Author: xushaocong
 # @Date: 2022-05-12 21:59:18
 # @LastEditTime: 2022-08-07 14:50:26
 # @LastEditors: xushaocong
 # @Description: 
 # @FilePath: /Cerberus-main/my_script/test.sh
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
python -u test.py test  -s 320 \
--resume /home/DISCOVER_summer2022/xusc/exp/Cerberus-main/networks/edge_cerberus8/checkpoints6/edge_final_slurm.pth.tar \
--batch-size 1 --workers 40 --gpu-ids "6" --run-id 0 --save-file "edge_final_slurm"\
2>&1 | tee -a logs/test.log






#* test all model under path 
# path=/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/networks/edge_cerberus8/checkpoints6/
# # idx=1;
# for model in $(ls $path); do 
#     echo $path$model;
#     model_name=(${model//./ });
#     echo ${model_name[0]}${model_name[1]};

#     python -u test.py test  -s 320 \
#     --resume $path$model \
#     --batch-size 1 --workers 40 --run-id 0 --save-file ${model_name[0]}${model_name[1]} \
#     2>&1 | tee -a logs/test.log
#     # idx=` expr $idx + 1 `;
# done;
# echo $idx;