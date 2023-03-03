
###
 # @Author: daniel
 # @Date: 2023-02-28 20:36:03
 # @LastEditTime: 2023-03-03 17:46:43
 # @LastEditors: daniel
 # @Description: 
 # @FilePath: /Cerberus-main/lib/matlab/eval/eval.sh
 # have a nice day
### 









# eval_dir=/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/cityscapes/cerberus/model_v1_ep106_1e-3_poly;



#* 验证best model (ep79, ~55) 是否真的优于 last model (ep 106)

python eval.py -d $eval_dir \
2>&1 | tee -a ../../../logs/eval_cityscapes.log