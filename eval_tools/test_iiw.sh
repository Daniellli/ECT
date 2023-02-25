
###
 # @Author: daniel
 # @Date: 2023-02-19 23:15:41
 # @LastEditTime: 2023-02-20 17:56:52
 # @LastEditors: daniel
 # @Description: 
 # @FilePath: /Cerberus-main/eval_tools/test_iiw.sh
 # have a nice day
### 




#* eval IIW

python eval_tools/iiw_evaluator.py --eval-data-dir '/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/networks/need2release/iiw_1' \
2>&1 | tee -a ./logs/eval_matlab.log
