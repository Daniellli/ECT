
###
 # @Author: daniel
 # @Date: 2023-02-28 20:36:03
 # @LastEditTime: 2023-03-10 17:28:04
 # @LastEditors: daniel
 # @Description: 
 # @FilePath: /Cerberus-main/lib/matlab/eval/eval.sh
 # have a nice day
### 







# eval_dir=/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/cityscapes/cerberus/model_v1_ep106_1e-3_poly;
# eval_dir=/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/sbd/cerberus/08:18:1677802680;


# eval_dir="/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/sbd/cerberus/edge_cerberus_with_dff_loss";
eval_dir="/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/sbd/cerberus/concat_decoder_feature";




python eval.py -d $eval_dir --dataset 'sbd' \
2>&1 | tee -a ../../../logs/eval_sbd.logs