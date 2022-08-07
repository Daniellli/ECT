'''
Author: xushaocong
Date: 2022-08-04 16:42:24
LastEditTime: 2022-08-04 16:49:46
LastEditors: xushaocong
Description: 
FilePath: /Cerberus-main/my_script/utils.py
email: xushaocong@stu.xmu.edu.cn
'''



import json
import os
from loguru import logger 


data= {
    "depth": {
        "ODS": "0.658",
        "OIS": "0.698",
        "AP": "0.640",
    },
    "normal": {
        "ODS": "0.454",
        "OIS": "0.514",
        "AP": "0.387",
    },
    "reflectance": {
        "ODS": "0.437",
        "OIS": "0.492",
        "AP": "0.350"
    },
    "illumination": {
        "ODS": "0.226",
        "OIS": "0.315",
        "AP": "0.142"
    }
}


 
ods = ois= ap=0
for k,v in data.items():
    ods+= float(v["ODS"])
    ois+=float(v["OIS"])
    ap+=float(v["AP"])


data["Average"] = {} 
data["Average"]["ODS"] = round( ods/4,3)
data["Average"]["OIS"] = round(ois/4,3)
data["Average"]["AP"] = round(ap/4,3)

logger.info(data['Average'])