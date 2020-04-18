
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn

from utils import Trigonometric_kernel, sample_pdf
from layers.RaySamplePoint import RaySamplePoint, RaySamplePoint_Near_Far
from .spacenet import SpaceNet
from layers.render_layer import VolumeRenderer, gen_weight
import time

class RFRender(nn.Module):

    def __init__(self, coarse_ray_sample, fine_ray_sample):
        super(RFRender, self).__init__()

        self.coarse_ray_sample = coarse_ray_sample
        self.fine_ray_sample = fine_ray_sample


        self.rsp_coarse = RaySamplePoint_Near_Far(self.coarse_ray_sample)
        #self.rsp_coarse = RaySamplePoint(self.coarse_ray_sample)

        self.spacenet = SpaceNet()
        self.spacenet_fine = SpaceNet()

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
    def forward(self, rays, bboxes, only_coarse = False):

        #if self.maxs is None:
        #    print('please set max_min before use.')
        #    return None


        #beg = time.time()
        #sampled_rays_coarse_t, sampled_rays_coarse_xyz  = self.rsp_coarse.forward(rays, bboxes)
        sampled_rays_coarse_t, sampled_rays_coarse_xyz  = self.rsp_coarse.forward(rays)

        sampled_rays_coarse_t = sampled_rays_coarse_t.detach()
        sampled_rays_coarse_xyz = sampled_rays_coarse_xyz.detach()

        
        rgbs, density = self.spacenet(sampled_rays_coarse_xyz, rays, self.maxs, self.mins)
        color_0, depth_0, acc_map_0, weights_0 = self.volume_render(sampled_rays_coarse_t, rgbs, density)

        #torch.cuda.synchronize()
        #print('render coarse:',time.time()-beg)

        if not only_coarse:


            z_samples = sample_pdf(sampled_rays_coarse_t.squeeze(), weights_0.squeeze()[...,1:-1], N_samples = self.fine_ray_sample)
            z_samples = z_samples.detach()   # (N,L)



            z_vals_fine, _ = torch.sort(torch.cat([sampled_rays_coarse_t.squeeze(), z_samples], -1), -1) #(N, L1+L2)
            samples_fine_xyz = z_vals_fine.unsqueeze(-1)*rays[:,:3].unsqueeze(1) + rays[:,3:].unsqueeze(1)  # (N,L1+L2,3)





            #beg = time.time()
            rgbs, density = self.spacenet_fine(samples_fine_xyz, rays, self.maxs, self.mins)
            color, depth, acc_map, weights = self.volume_render(z_vals_fine.unsqueeze(-1), rgbs, density)

        else:
            color, depth, acc_map, weights = color_0, depth_0, acc_map_0, weights_0
        #torch.cuda.synchronize()
        #print('render fine:',time.time()-beg)
         #print('depth range:',depth.max(),depth.min())
        #print('color range:',color.max(),color.min())


        return (color, depth, acc_map) , (color_0, depth_0, acc_map_0)


    def set_max_min(self, maxs, mins):
        self.maxs = maxs
        self.mins = mins