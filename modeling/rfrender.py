
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn

from utils import Trigonometric_kernel
from layers.RaySamplePoint import RaySamplePoint
from .spacenet import SpaceNet
from layers.render_layer import VolumeRenderer

class RFRender(nn.Module):

    def __init__(self):
        super(RFRender, self).__init__()


        self.rsp_coarse = RaySamplePoint()

        self.spacenet = SpaceNet()

        self.volume_render = VolumeRenderer()

        self.maxs = None
        self.mins = None



    '''
    INPUT

    rays: rays  (N,6)
    bboxes: bounding boxes (N,8,3)

    OUTPUT

    rgbs: color of each ray (N,3) 
    depths:  depth of each ray (N,1) 

    '''
    def forward(self, rays, bboxes):

        if self.maxs is None:
            print('please set max_min before use.')
            return None

        sampled_rays_coarse_t, sampled_rays_coarse_xyz  = self.rsp_coarse.forward(rays, bboxes, method=None)

        rgbs, density = self.spacenet(sampled_rays_coarse_xyz, rays, self.maxs, self.mins)




        color, depth = self.volume_render(sampled_rays_coarse_t, rgbs, density)

         #print('depth range:',depth.max(),depth.min())
        #print('color range:',color.max(),color.min())


        return color, depth


    def set_max_min(self, maxs, mins):
        self.maxs = maxs
        self.mins = mins