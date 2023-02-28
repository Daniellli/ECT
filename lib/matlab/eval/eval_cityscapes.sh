
###
 # @Author: daniel
 # @Date: 2023-02-26 16:00:56
 # @LastEditTime: 2023-02-28 20:35:44
 # @LastEditors: daniel
 # @Description: 
 # @FilePath: /Cerberus-main/lib/matlab/eval/eval_cityscapes.sh
 # have a nice day
### 


#* Cityscapes
matlab -batch demoBatchEvalCityscapes \
2>&1 | tee -a ../../../logs/eval_cityscapes.log




#* SBD
# matlab -batch demoBatchEvalSBD