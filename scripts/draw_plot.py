'''
Author: xushaocong
Date: 2022-07-25 15:33:47
LastEditTime: 2022-07-25 15:55:38
LastEditors: xushaocong
Description: 
FilePath: /Cerberus-main/my_script/draw_plot.py
email: xushaocong@stu.xmu.edu.cn
'''



import os


import wandb
from loguru import logger 



import glob


from os.path import join, split

import json

path = "/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/networks/lr@1e-05_ep@300_bgw@1.0_rindw@1.0_1658548858"





all_data= sorted(glob.glob(join(path,'model_res*')))
# run = wandb.init(project="draw_edge_cerberus")
# run.name= "draw_edge_cerberus_"+run.name


best_ods_name =best_ois_name =best_ap_name = None

best_ods = best_ois = best_ap  =  0

for x in all_data:

    with open(join(x,"eval_res.json"), 'r')as f :
        eval_res = json.load(f)
        
    # wandb.log(eval_res["Average"])
    if eval_res["Average"]["ODS"] > best_ods:
        best_ods = eval_res["Average"]["ODS"]
        best_ods_name = split(x)[-1]
        
    
    if eval_res["Average"]["OIS"] > best_ois:
        best_ois = eval_res["Average"]["OIS"]
        best_ois_name = split(x)[-1]

    if eval_res["Average"]["AP"] > best_ap:
        best_ap = eval_res["Average"]["AP"]
        best_ap_name = split(x)[-1]
    
    
    
print(f"ODS : {best_ods}  " +best_ods_name)
print(f"OIS: {best_ois}  "   +best_ois_name)
print(f"AP : {best_ap}   "  +best_ap_name)





