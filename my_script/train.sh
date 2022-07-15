###
 # @Author: xushaocong
 # @Date: 2022-05-12 21:59:29
 # @LastEditTime: 2022-07-15 14:02:28
 # @LastEditors: xushaocong
 # @Description: 
 # @FilePath: /cerberus/my_script/train.sh
 # email: xushaocong@stu.xmu.edu.cn
### 

#* 原来的训练代码
# python   -u  main.py \
# train   -s 512 --batch-size 1 --random-scale 2 \
# --random-rotate 10 --epochs 5 --lr 0.007 --momentum 0.9 \
# --lr-mode poly --workers 12 --classes 1 \
# 2>&1 | tee -a logs/train.log



#* 进一步封装 DPT
# python   main4.py train  -s 320 --batch-size 12  --epochs 1 --lr 1e-4 --momentum 0.9 \
# --lr-mode poly --workers 12 --gpu-ids '2,4,5' \
# 2>&1 | tee -a logs/train.log



#* 炼丹代码
lr=1e-5;
batch_size=12;
# gpuids="0,1,2,3,4,5,6,7";
gpuids="0,1,2,3,4,5";
gpu_number=6;
epoch=300;
bg_weights=(1);
rind_weights=(1);

for idx in $(seq 0 1 0);do 
    echo bg_weights = ${bg_weights[$idx]} ,rind_weights = ${rind_weights[$idx]};
    # python   train.py train  -s 320 --batch-size $batch_size  --epochs $epoch --lr $lr --momentum 0.9 \
    #     --lr-mode poly --workers 12 --gpu-ids $gpuids --bg-weight ${bg_weights[$idx]} --rind-weight ${rind_weights[$idx]} \
    #     2>&1 | tee -a logs/train.log

    #*========================================================================================
    python  -m torch.distributed.launch --nproc_per_node=$gpu_number   --master_port 29506 \
        train2.py train  -s 320 --batch-size $batch_size  --epochs $epoch --lr $lr --momentum 0.9 \
        --lr-mode poly --workers 12 --gpu-ids $gpuids --bg-weight ${bg_weights[$idx]} --rind-weight ${rind_weights[$idx]} \
        2>&1 | tee -a logs/train.log
    #*=======================================================git a=================================

    #*========================================================================================resume 
    #* --resume 绝对路径和 相对路径都可以 
    # python  -m torch.distributed.launch --nproc_per_node=$gpu_number   --master_port 29506 \
    #     train2.py train  -s 320 --batch-size $batch_size  --epochs $epoch --lr $lr --momentum 0.9 \
    #     --resume "/data3/xusc/exp/cerberus/networks/lr@1e-05_ep@300_bgw@1.0_rindw@1.0_1657761722/checkpoints/ckpt_rank000_ep0100.pth.tar" \
    #     --lr-mode poly --workers 12 --gpu-ids $gpuids --bg-weight ${bg_weights[$idx]} --rind-weight ${rind_weights[$idx]} \
    #     --save-dir "/data3/xusc/exp/cerberus/networks/lr@1e-05_ep@300_bgw@1.0_rindw@1.0_1657761722/checkpoints/" \
    #     2>&1 | tee -a logs/train.log
    #*========================================================================================

done



#* torch.distributed.launch 实现分布式
# python  -m torch.distributed.launch --nproc_per_node=$gpu_number   --master_port 29506 \
# train2.py train  -s 320 --batch-size $batch_size  --epochs $epoch --lr $lr --momentum 0.9 \
# --lr-mode poly --workers 12 --gpu-ids $gpuids \
# 2>&1 | tee -a logs/train.log






