
###
 # @Author: daniel
 # @Date: 2023-02-06 20:17:43
 # @LastEditTime: 2023-08-14 11:00:20
 # @LastEditors: daniel
 # @Description: 
 # @FilePath: /Cerberus-main/scripts/train_ect.sh
 # have a nice day
### 




conda activate ect
gpuids="2";
gpu_number=1;


CUDA_VISIBLE_DEVICES=$gpuids python  -m torch.distributed.launch --nproc_per_node=$gpu_number   --master_port 29510 \
ect_trainer.py train  -s 320 --batch-size 4  --epochs 300 --lr 1e-5 \
--lr-mode poly --workers 16 --gpu-ids $gpuids --bg-weight 0.5 --rind-weight 1 \
--extra-loss-weight 1e+3 --edge-loss-gamma 0.3 --edge-loss-beta 1 \
--rind-loss-gamma 0.3  --rind-loss-beta 5 --cause-token-num 8 \
2>&1 | tee -a logs/train.log
 
# --wandb


