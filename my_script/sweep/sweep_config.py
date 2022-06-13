'''
Author: xushaocong
Date: 2022-05-07 15:08:33
LastEditTime: 2022-06-13 20:58:35
LastEditors: xushaocong
Description: 
FilePath: /Cerberus-main/my_script/sweep/sweep_config.py
email: xushaocong@stu.xmu.edu.cn
'''


# define a sweep dictionary containing all the hyperparameters
sweep_config = {
    "name": "train_cerberus",
    'method': 'grid',  # grid, random
    # 'metric': {
    #     'name': 'loss',
    #     'goal': 'minimize'
    # },
    'parameters': {

        'epochs': {
            'values': [300]
        },
        'phase': {
            'values': ["train"]
        },
        'batch_size': {
            'values': [1]
        },
         'gpuids': {
            'values': ["6,7"]
        },
        'learning_rate': {
            'values': [1e-3,1e-4,1e-5,1e-6,1e-7,1e-8]
        },
    }
}



