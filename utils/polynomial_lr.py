'''
Author: daniel
Date: 2023-03-02 22:11:02
LastEditTime: 2023-03-02 22:27:55
LastEditors: daniel
Description: 
FilePath: /cerberus/utils/polynomial_lr.py
have a nice day
'''




class PolynomialLR:
    
    def __init__(self,optimizer,total_iters,power=1.0,last_epoch=-1):
        self.optimizer = optimizer
        self.orgin_lr = self.optimizer.param_groups.copy()
        self.total_iters = total_iters
        self.power = power
        

    def step(self,epoch):
        for idx,param_group in enumerate(self.orgin_lr):
            self.optimizer.param_groups[idx]['lr'] = param_group['lr'] * pow((1 - epoch / self.total_iters),self.power)
    
        
    def get_last_lr(self):
        return [parameter['lr'] for parameter in self.optimizer.param_groups]
    
    
    def load_state_dict(self,state_dict):
        self.__dict__.update(state_dict)
        
    def state_dict(self):
         return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}
        

    
    
    