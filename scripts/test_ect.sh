###
 # @Author:   "  "
 # @Date: 2022-05-12 21:59:18
 # @LastEditTime: 2023-07-26 15:04:35
 # @LastEditors: daniel
 # @Description: 
 # @FilePath: /Cerberus-main/scripts/test_ect.sh
 # email:  
### 


source /usr/local/miniconda3/etc/profile.d/conda.sh 

conda activate cerberus2





gpuids='6'
export CUDA_VISIBLE_DEVICES=$gpuids;






#* test one model 
resume_model="/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/networks/need2release/checkpoints/full_version.pth.tar";



python -u test.py test  -s 320 --resume $resume_model \
--batch-size 1 --workers 40 --gpu-ids $gpuids \
2>&1 | tee -a logs/test_iiw.log










# for i in {250..300}; do
#     if (( ($i - 250) % 5 == 0 )); then
#         resume_model="/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/networks/2023-05-26-13:04:1685077468#CN20/checkpoints/ckpt_ep0${i}.pth.tar";
#         echo $resume_model;
#         python  -m torch.distributed.launch --nproc_per_node=1 \
#         --master_port 32654 ect_trainer.py test  -s 320 \
#         --resume $resume_model --cause-token-num 3 \
#         --batch-size 1 --workers 40 --gpu-ids $gpuids --cause-token-num 20 \
#         2>&1 | tee -a logs/test.log

#     fi
# done











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