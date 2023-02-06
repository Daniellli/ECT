###
 # @Author: xushaocong
 # @Date: 2022-09-05 23:15:06
 # @LastEditTime: 2022-09-23 10:57:27
 # @LastEditors: xushaocong
 # @Description: 
 # @FilePath: /Cerberus-main/my_script/inference.sh
 # email: xushaocong@stu.xmu.edu.cn
### 



#* 模型路径
#* 要inference 的文件夹路径


python -u inference.py --resume /home/DISCOVER_summer2022/xusc/exp/Cerberus-main/networks/need2release/checkpoints/full_version.pth.tar \
--data-dir /home/DISCOVER_summer2022/xusc/exp/Cerberus-main/plot/demo/robotics/imgs \
--batch-size 1  --gpu-ids "1" --run-id 0 --workers 1 2>&1 | tee -a logs/test.log




# for x in $(seq 0 1 3);do 
# name=`printf '%04d' $x`;
# echo $name;
# python -u inference.py --resume /home/DISCOVER_summer2022/xusc/exp/Cerberus-main/networks/need2release/checkpoints/full_version.pth.tar \
# --data-dir /home/DISCOVER_summer2022/xusc/exp/Cerberus-main/plot/demo/$name/imgs \
# --batch-size 1  --gpu-ids "0" --run-id 0 --workers 1 2>&1 | tee -a logs/test.log
# done;