###
 # @Author: xushaocong
 # @Date: 2022-06-13 15:54:14
 # @LastEditTime: 2022-06-13 18:02:04
 # @LastEditors: xushaocong
 # @Description: 
 # @FilePath: /Cerberus-main/eval_tools/test.sh
 # email: xushaocong@stu.xmu.edu.cn
### 




cd eval_tools;
output_dir=$1;
# source activate
# conda deactivate
# conda activate matlab #* 不需要切换环境也可以测试
# echo "output_dir == $output_dir";
python test.py -d $output_dir;





