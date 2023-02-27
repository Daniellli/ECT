
###
 # @Author: daniel
 # @Date: 2023-02-26 16:00:56
 # @LastEditTime: 2023-02-26 16:11:13
 # @LastEditors: daniel
 # @Description: 
 # @FilePath: /rindnet/lib/matlab/eval/eval_cityscapes.sh
 # have a nice day
### 



matlab -batch demoBatchEvalCityscapes \
2>&1 | tee -a ../../../logs/eval_cityscapes.log