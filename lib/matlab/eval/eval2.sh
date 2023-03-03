
###
 # @Author: daniel
 # @Date: 2023-02-28 20:36:03
 # @LastEditTime: 2023-03-03 12:22:25
 # @LastEditors: daniel
 # @Description: 
 # @FilePath: /Cerberus-main/lib/matlab/eval/eval2.sh
 # have a nice day
### 









eval_dir=/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/cityscapes/cerberus/model_v1_ep45_poly_8e-3;


#* 验证best model  of v100; with bs =1 

python eval.py -d $eval_dir \
2>&1 | tee -a ../../../logs/eval_cityscapes.log