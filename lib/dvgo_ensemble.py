import os
import time
import functools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import segment_coo




'''Model'''
class DirectVoxGoEnsemble(torch.nn.Module):

    def __init__(self, models) -> None:
        super().__init__()
        self.models = nn.ModuleList([m for m in models])

        
    def forward(self, rays_o, rays_d, viewdirs, global_step=None, **render_kwargs):
        render_result_list = []

        for model in self.models:
            render_result = model(
                    rays_o, rays_d, viewdirs,
                    global_step=global_step, is_train=True,
                    **render_kwargs)
            render_result_list.append(render_result)
        

        return render_result_list