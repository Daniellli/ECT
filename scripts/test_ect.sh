###
 # @Author:   "  "
 # @Date: 2022-05-12 21:59:18
 # @LastEditTime: 2023-08-14 10:57:17
 # @LastEditors: daniel
 # @Description: 
 # @FilePath: /Cerberus-main/scripts/test_ect.sh
 # email:  
### 



conda activate ect
gpuids='1'
resume_model="XXX";

export CUDA_VISIBLE_DEVICES=$gpuids python  -m torch.distributed.launch --nproc_per_node=1 \
--master_port $RANDOM ect_trainer.py test  -s 320 \
--resume $resume_model \
--batch-size 1 --workers 40 --gpu-ids $gpuids --cause-token-num 4 \
2>&1 | tee -a logs/test.log

