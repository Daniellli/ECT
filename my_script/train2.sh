
###
 # @Author: xushaocong
 # @Date: 2022-07-21 22:57:50
 # @LastEditTime: 2022-07-22 20:10:27
 # @LastEditors: xushaocong
 # @Description: 
 # @FilePath: /Cerberus-main/my_script/train2.sh
 # email: xushaocong@stu.xmu.edu.cn
### 

#* 炼丹代码
lr=1e-5;

batch_size=24;
gpuids="2,3,4,5,6,7";
gpu_number=6;
epoch=300;
bg_weights=1;
rind_weights=1;
extra_loss_weight=(200 300 500);

for idx in $(seq 1 1 2);do 
    echo ${extra_loss_weight[$idx]};
    #*========================================================================================
    python  -m torch.distributed.launch --nproc_per_node=$gpu_number   --master_port 29506 \
        train2.py train  -s 320 --batch-size $batch_size  --epochs $epoch --lr $lr --momentum 0.9 \
        --lr-mode poly --workers 12 --gpu-ids $gpuids --bg-weight $bg_weights --rind-weight $rind_weights \
        --extra-loss-weight ${extra_loss_weight[$idx]}  --validation  2>&1 | tee -a logs/train.log
    #*=======================================================git a=================================

done

