###
 # @Author: xushaocong
 # @Date: 2022-06-13 15:54:14
 # @LastEditTime: 2022-08-09 13:21:08
 # @LastEditors: xushaocong
 # @Description: 
 # @FilePath: /Cerberus-main/eval_tools/test.sh
 # email: xushaocong@stu.xmu.edu.cn
### 




cd eval_tools;
#* 这个如果是1 就test,如果是2 就不测试
echo param == $2,dir == $1;
# source activate
# conda deactivate
# conda activate matlab #* 不需要切换环境也可以测试
# echo "output_dir == $output_dir";
#! 注意空格的问题
# if [ $2=="1" ]; then 
#     echo  "test edge ";
#     python test.py -d $1 --test-edge;
# else
#     echo  "do not test edge ";
#     python test.py -d $1;
# fi;

# python test.py -d $1;
python test.py -d $1 --test-edge;






