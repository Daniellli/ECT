###
 # @Author:   "  "
 # @Date: 2022-06-13 15:54:14
 # @LastEditTime: 2023-05-24 14:41:19
 # @LastEditors: daniel
 # @Description: 
 # @FilePath: /Cerberus-main/eval_tools/test.sh
 # email:  
### 




cd eval_tools;
#* 这个如果是1 就test,如果是2 就不测试
echo  dir == $1,param == $2;
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





source /usr/local/miniconda3/etc/profile.d/conda.sh 


conda activate cerberus2

# python test.py -d $1
# python test.py -d $1 --test-edge;

# python test.py --eval-data-dir "/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/networks/2023-03-16-15:10:1678950646"  --dataset BSDS-RIND
python test.py --eval-data-dir $1  --dataset BSDS-RIND


# * SBU
# python test.py --eval-data-dir /home/DISCOVER_summer2022/xusc/exp/Cerberus-main/networks/need2release/SBU_0 \
# --dataset SBU 2>&1 | tee -a ../logs/eval_matlab_sbu.log



# python test.py --eval-data-dir /home/DISCOVER_summer2022/xusc/exp/Cerberus-main/networks/need2release/ISTD_0 \
# --dataset ISTD 2>&1 | tee -a ../logs/eval_matlab_istd.log



# python test.py --eval-data-dir '/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/networks/need2release/nyud2_1' \
# --dataset NYUD2 2>&1 | tee -a ../logs/eval_matlab.log





#! GT, OURS, RINDNET,DFF,RCF,*OFNet,HED 
# path=/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/networks/precomputed/
# paths=(dff rcf hed ofnet);
# for model in ${paths[@]} ;do 
# echo $path$model;
# python test.py -d $path$model --test-edge;
# done;












