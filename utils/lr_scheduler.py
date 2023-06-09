'''
Author: daniel
Date: 2023-03-01 18:18:57
LastEditTime: 2023-03-02 22:38:37
LastEditors: daniel
Description: 
FilePath: /cerberus/utils/lr_scheduler.py
have a nice day
'''
# ------------------------------------------------------------------------
# Group-Free
# Copyright (c) 2021 Ze Liu. All Rights Reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------
# noinspection PyProtectedMember
from torch.optim.lr_scheduler import _LRScheduler, MultiStepLR, CosineAnnealingLR,ReduceLROnPlateau
from .polynomial_lr import PolynomialLR



from IPython import embed 



# noinspection PyAttributeOutsideInit
class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
      Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
      Args:
          optimizer (Optimizer): Wrapped optimizer.
          multiplier: init learning rate = base lr / multiplier
          warmup_epoch: target learning rate is reached at warmup_epoch, gradually
          after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
      """

    def __init__(self, optimizer, multiplier, warmup_epoch, after_scheduler, last_epoch=-1):
        self.multiplier = multiplier
        if self.multiplier <= 1.:
            raise ValueError('multiplier should be greater than 1.')
        self.warmup_epoch = warmup_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        if self.last_epoch > self.warmup_epoch:
            return self.after_scheduler.get_lr()
        else:
            return [base_lr / self.multiplier * ((self.multiplier - 1.) * self.last_epoch / self.warmup_epoch + 1.)
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        if epoch > self.warmup_epoch:
            self.after_scheduler.step(epoch - self.warmup_epoch)
        else:
            super(GradualWarmupScheduler, self).step(epoch)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """

        state = {key: value for key, value in self.__dict__.items() if key != 'optimizer' and key != 'after_scheduler'}
        state['after_scheduler'] = self.after_scheduler.state_dict()
        return state

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """

        after_scheduler_state = state_dict.pop('after_scheduler')
        self.__dict__.update(state_dict)
        self.after_scheduler.load_state_dict(after_scheduler_state)


def get_scheduler(optimizer, n_iter_per_epoch, args):
    if "cosine" in args.lr_scheduler:
        """
            not used parameter : 
                1. lr_decay_rate
                2.lr_decay_epochs
                
            hyperparameter meaning:
                1. T_max: the step number optimizer need to step. 
        """
        scheduler = CosineAnnealingLR(
            optimizer=optimizer,
            eta_min=0.000001, #* the  minimun learning rate , 0 by default
            #*  change the warmup_epoch to -1 as I have not  warm up epoch
            T_max=(args.epochs +1) * n_iter_per_epoch) 
    elif "step" == args.lr_scheduler:
        if isinstance(args.lr_decay_epochs, int):
            args.lr_decay_epochs = [args.lr_decay_epochs]
            
        scheduler = MultiStepLR(
                optimizer=optimizer,
                gamma=args.lr_decay_rate,
                #* milestones=[(m - args.warmup_epoch) * n_iter_per_epoch for m in args.lr_decay_epochs]), change the warmup_epoch to -1 as i have not  warm up epoch
                milestones=[(m +1) * n_iter_per_epoch for m in args.lr_decay_epochs],
                # verbose=True
            )
        print('scheduler step counter : ',scheduler._step_count,'last epoch : ',scheduler.last_epoch,'milestones',scheduler.milestones)
        
    elif 'poly' in args.lr_scheduler:
        """
            not used parameter : 
                1. lr_decay_rate
                2.lr_decay_epochs
            lr_decay_rate should be set as 0.9
        """
        
        scheduler = PolynomialLR(optimizer = optimizer,
                                total_iters = args.epochs,
                                power = args.lr_decay_rate) #* 
        
        
        print('PolynomialLR schedule  total epoch :',args.epochs,' power : ',args.lr_decay_rate)
        
    # elif "step2" == args.lr_scheduler :
    #     scheduler = StepLR(optimizer, step_size=1, gamma=args.lr_decay_rate)
    #     print('scheduler step counter : ',scheduler._step_count,'last epoch : ',
    #             scheduler.last_epoch)
    else:
        raise NotImplementedError(f"scheduler {args.lr_scheduler} not supported")

    # if args.warmup_epoch > 0:
    #     scheduler = GradualWarmupScheduler(
    #         optimizer,
    #         multiplier=args.warmup_multiplier,
    #         after_scheduler=scheduler,
    #         warmup_epoch=args.warmup_epoch * n_iter_per_epoch)
    return scheduler
