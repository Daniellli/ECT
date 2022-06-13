'''
Author: xushaocong
Date: 2022-06-13 10:30:59
LastEditTime: 2022-06-13 13:04:36
LastEditors: xushaocong
Description:  使用matlab engin 进行eval
FilePath: /Cerberus-main/eval_tools/test.py
email: xushaocong@stu.xmu.edu.cn
'''




import matlab
import matlab.engine
import argparse
import os 
import os.path as osp
import sys
import json 
# parser = argparse.ArgumentParser(description='')
# parser.add_argument('-d', '--eval-data-dir', 
#     default='./dataset/BSDS_RIND_mine',help="eval data dir  , must be absolution dir ")
# args = parser.parse_args()
sys.path.append(osp.dirname(__file__)) #* 加这行无效!
# os.chdir(osp.dirname(__file__)) #* 需要到当前目录才能执行??? 


'''
description:  调用matlab 来eval , 
param {*} eval_data_dir : 成功在绝对路径进行测试, 相对路径还未测试
return {*}
'''
def test_by_matlab(eval_data_dir):
    eng = matlab.engine.start_matlab()
    eval_res = eng.eval_edge(eval_data_dir) #* 评估完会返回一串 string 
    keys=['depth','normal','reflectance','illumination']
    res = {}
    for idx, eval_value in enumerate(eval_res):#* ODS, OIS, AP, R50        
        res[keys[idx]] = {"ODS": "%.3f"%(eval_value[0]),"OIS":  "%.3f"%(eval_value[1]),"AP": "%.3f"%(eval_value[2])}
    with open (osp.join(eval_data_dir,"eval_res.json"),'w')as f :
        json.dump(res,f)
    return res


if __name__ =="__main__":
    # test_by_matlab("/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/networks/model_res")
    test_by_matlab("/home/DISCOVER_summer2022/xusc/exp/Cerberus-main/networks/dashing-wind-713/model_res")

    





