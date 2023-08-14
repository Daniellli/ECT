###
 # @Author:   "  "
 # @Date: 2022-06-13 15:54:14
 # @LastEditTime: 2023-08-14 10:44:47
 # @LastEditors: daniel
 # @Description: eval BSDS-RIND, NYUD2, SBU,ISTD...
 # @FilePath: /Cerberus-main/eval_tools/test.sh
 # email:  
### 




cd eval_tools;

source /usr/local/miniconda3/etc/profile.d/conda.sh 
conda activate cerberus2



#? test generic edge or not 
# echo  dir == $1,param == $2;
# echo "output_dir == $output_dir";
# if [ $2=="1" ]; then 
#     echo  "test edge ";
#     python test.py -d $1 --test-edge;
# else
#     echo  "do not test edge ";
#     python test.py -d $1;
# fi;





#*=========================test BSDS-RIND=========================
# python test.py -d $1
# python test.py -d $1 --test-edge;


python test.py --eval-data-dir "/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/networks/need2release/300"  --dataset BSDS-RIND

# python test.py --eval-data-dir $1  --dataset BSDS-RIND --test-edge

#*================================================================





#*=========================test SBU================================
# python test.py --eval-data-dir /home/DISCOVER_summer2022/xusc/exp/Cerberus-main/networks/need2release/SBU_0 \
# --dataset SBU 2>&1 | tee -a ../logs/eval_matlab_sbu.log

#*================================================================



#*=========================test ISTD================================
# python test.py --eval-data-dir /home/DISCOVER_summer2022/xusc/exp/Cerberus-main/networks/need2release/ISTD_0 \
# --dataset ISTD 2>&1 | tee -a ../logs/eval_matlab_istd.log

#*================================================================



#*=========================test NYUD2================================
# python test.py --eval-data-dir '/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/networks/need2release/nyud2_1' \
# --dataset NYUD2 2>&1 | tee -a ../logs/eval_matlab.log

#*================================================================



#! GT, OURS, RINDNET,DFF,RCF,*OFNet,HED 
# path=/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/networks/precomputed/
# paths=(dff rcf hed ofnet);
# for model in ${paths[@]} ;do 
# echo $path$model;
# python test.py -d $path$model --test-edge;
# done;












