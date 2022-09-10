###
 # @Author: xushaocong
 # @Date: 2022-09-06 13:16:08
 # @LastEditTime: 2022-09-06 14:01:25
 # @LastEditors: xushaocong
 # @Description: 
 # @FilePath: /Cerberus-main/eval_tools/inference.sh
 # email: xushaocong@stu.xmu.edu.cn
### 



cd eval_tools;
python test.py -d $1 --inference-only;