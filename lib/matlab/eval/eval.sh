
###
 # @Author: daniel
 # @Date: 2023-02-28 20:36:03
 # @LastEditTime: 2023-03-01 16:42:31
 # @LastEditors: daniel
 # @Description: 
 # @FilePath: /Cerberus-main/lib/matlab/eval/eval.sh
 # have a nice day
### 









# eval_dir=/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/cityscapes/cerberus/11:49:1677556146;
# eval_dir=/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/cityscapes/cerberus/model_v2_prediction;
# eval_dir=/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/cityscapes/cerberus/model_v1_ep81;
# eval_dir=/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/cityscapes/cerberus/model_v2_ep87;
eval_dir=/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/cityscapes/cerberus/model_v1_ep96;





python eval.py -d $eval_dir \
2>&1 | tee -a ../../../logs/eval_cityscapes.log