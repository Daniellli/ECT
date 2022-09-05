###
 # @Author: xushaocong
 # @Date: 2022-09-05 23:15:06
 # @LastEditTime: 2022-09-05 23:35:46
 # @LastEditors: xushaocong
 # @Description: 
 # @FilePath: /Cerberus-main/my_script/inference.sh
 # email: xushaocong@stu.xmu.edu.cn
### 



python -u reference.py --resume /home/DISCOVER_summer2022/xusc/exp/Cerberus-main/networks/need2release/checkpoints/full_version.pth.tar \
--data-dir /home/DISCOVER_summer2022/xusc/exp/Cerberus-main/plot-rind-edge-pr-curves/demo/Transform_video_Q15-Img \
--batch-size 1  --gpu-ids "0" --run-id 0  2>&1 | tee -a logs/test.log

