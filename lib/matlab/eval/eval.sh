









# eval_dir=/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/cityscapes/cerberus/11:49:1677556146;
eval_dir=/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/cityscapes/cerberus/model_v2_prediction;



python eval.py -d $eval_dir \
2>&1 | tee -a ../../../logs/eval_cityscapes.log