



class PolynomialLR:
    
    def __init__(self,optimizer,total_iters,power=1.0,last_epoch=-1):
        self.optimizer = optimizer
        self.orgin_lr = self.optimizer.param_groups.copy()
        self.total_iters = total_iters
        self.power = power
        

    def step(self,epoch):
        for idx,param_group in enumerate(self.orgin_lr):
            self.optimizer.param_groups[idx]['lr'] = param_group['lr'] * \
                    (1 - epoch / self.total_iters) ** self.power
    
        
    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']
    
    
    def load_state_dict(self,state_dict):
        self.__dict__.update(state_dict)
        
    def state_dict(self):
         return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}
        

    
    
    