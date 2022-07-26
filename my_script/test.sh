###
 # @Author: xushaocong
 # @Date: 2022-05-12 21:59:18
 # @LastEditTime: 2022-07-25 16:48:31
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




for i in $(seq 260 5 299); do 
    a=$(printf "%04d" $i);
    
    model_path=/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/networks/lr@1e-05_ep@300_bgw@1.0_rindw@1.0_1658548858/checkpoints/ckpt_rank000_ep$a.pth.tar

    echo $model_path,$i;

    python -u test.py test  -s 320 \
    --resume $model_path \
    --batch-size 1 --workers 40 --gpu-ids "6" --run-id $i \
    2>&1 | tee -a logs/test.log

done;


#* test one model 
# python -u test.py test  -s 320 \
# --resume /home/DISCOVER_summer2022/xusc/exp/Cerberus-main/networks/lr@1e-05_ep@300_bgw@1.0_rindw@1.0_1658548858/checkpoints/ckpt_rank000_ep0299.pth.tar \
# --batch-size 1 --workers 40 --gpu-ids "2" --run-id 2 \
# 2>&1 | tee -a logs/test.log



#* test all model under path 
# path="/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/networks/new_model2/checkpoints/"
# idx=1;
# for model in $(ls $path); do 
#     echo $path$model, $idx;

#     python -u test.py test  -s 320 \
#     --resume $path$model \
#     --batch-size 1 --workers 40 --run-id $idx \
#     2>&1 | tee -a logs/test.log
#     idx=` expr $idx + 1 `;
# done;
# echo $idx;