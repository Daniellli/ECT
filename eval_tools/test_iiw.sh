
###
 # @Author: daniel
 # @Date: 2023-02-19 23:15:41
 # @LastEditTime: 2023-08-06 21:50:18
 # @LastEditors: daniel
 # @Description: eval the IIW predicitons
 # @FilePath: /Cerberus-main/eval_tools/test_iiw.sh
 # have a nice day
### 



source  /usr/local/miniconda3/etc/profile.d/conda.sh 
conda activate cerberus2


python eval_tools/iiw_evaluator.py --eval-data-dir '/home/DISCOVER_summer2022/xusc/exp/EDTER/logs/IIW' \
2>&1 | tee -a ./logs/eval_matlab.log


