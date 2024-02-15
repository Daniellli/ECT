###
 # @Author:   "  "
 # @Date: 2022-05-12 21:59:18
 # @LastEditTime: 2023-12-22 10:09:49
 # @LastEditors: daniel
 # @Description: 
 # @FilePath: /Cerberus-main/scripts/test_ect.sh
 # email:  
### 



# conda activate ect
source /usr/local/miniconda3/etc/profile.d/conda.sh 
conda activate cerberus2
gpuids='1'
# resume_model="networks/2023-12-20-13:32:1703050351#CN4/checkpoints/ckpt_ep0255.pth.tar";
resume_model="/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/networks/2023-12-20-13:32:1703050351#CN4/checkpoints/model_best.pth.tar";

# python -c 'import torch;print(torch.cuda.is_available())'


CUDA_VISIBLE_DEVICES=$gpuids  python -m torch.distributed.launch --nproc_per_node=1 \
--master_port $RANDOM ect_trainer.py test  -s 320 \
--resume $resume_model \
--batch-size 1 --workers 40 --gpu-ids $gpuids --cause-token-num 4 \
2>&1 | tee -a logs/test.log

