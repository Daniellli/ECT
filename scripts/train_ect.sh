
###
 # @Author: daniel
 # @Date: 2023-02-06 20:17:43
 # @LastEditTime: 2024-01-03 19:11:09
 # @LastEditors: daniel
 # @Description: 
 # @FilePath: /Cerberus-main/scripts/train_ect.sh
 # have a nice day
### 


source /usr/local/miniconda3/etc/profile.d/conda.sh 
conda activate cerberus2
# conda activate ect
gpuids="0,1,2";
gpu_number=3;


CUDA_VISIBLE_DEVICES=$gpuids python  -m torch.distributed.launch --nproc_per_node=$gpu_number --master_port 29510 \
ect_trainer.py train  -s 320 --batch-size 4  --epochs 300 --lr 1e-5 \
--lr-mode poly --workers 16 --gpu-ids $gpuids --bg-weight 0.5 --rind-weight 1 \
--extra-loss-weight 1e+3 --edge-loss-gamma 0.3 --edge-loss-beta 1 \
--rind-loss-gamma 0.3  --rind-loss-beta 5 --cause-token-num 4 --wandb \
2>&1 | tee -a logs/train.log
 



