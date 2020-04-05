
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


        self. rsp_coarse = RaySamplePoint()

        self.spacenet = SpaceNet()

        self.volume_render = VolumeRenderer()



    '''
    INPUT

    rays: rays  (N,6)
    bboxes: bounding boxes (N,8,3)

    OUTPUT

    rgbs: color of each ray (N,3) 
    depths:  depth of each ray (N,1) 

    '''
    def forward(self, rays, bboxes):

        sampled_rays_coarse_xyz, sampled_rays_coarse_t  = rsp.forward(rays, bboxes, method=None)


        rgbs, density = self.spacenet(sampled_rays_coarse_xyz, rays)


        color, depth = self.volume_render(sampled_rays_coarse_t, rgbs, density)


        return color, depth