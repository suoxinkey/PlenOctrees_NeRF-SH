# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import torch


def make_optimizer(cfg, model):

    weight_decay = cfg.SOLVER.WEIGHT_DECAY
    lr = cfg.SOLVER.BASE_LR


    if cfg.SOLVER.OPTIMIZER_NAME == 'Adam':
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, betas=(0.9, 0.999),weight_decay =weight_decay )
    else:
        pass

    return optimizer
