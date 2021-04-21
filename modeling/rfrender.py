
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

    def __init__(self, coarse_ray_sample, fine_ray_sample, boarder_weight, sample_method = 'NEAR_FAR', same_space_net = False, TriKernel_include_input = True, use_sh=True):
        super(RFRender, self).__init__()

        self.coarse_ray_sample = coarse_ray_sample
        self.fine_ray_sample = fine_ray_sample
        self.sample_method = sample_method

        #self.rsp_coarse = RaySamplePoint_Near_Far(self.coarse_ray_sample)
        if self.sample_method == 'NEAR_FAR':
            self.rsp_coarse = RaySamplePoint_Near_Far(self.coarse_ray_sample)   # use near far to sample points on rays
        else:
            self.rsp_coarse = RaySamplePoint(self.coarse_ray_sample)            # use bounding box to define point sampling ranges on rays

        self.spacenet = SpaceNet(include_input = TriKernel_include_input, use_sh = use_sh)
        self.spacenet_fine = SpaceNet(include_input = TriKernel_include_input, use_sh = use_sh)

        self.volume_render = VolumeRenderer(use_mask= True, boarder_weight = boarder_weight)




    '''
    INPUT

    rays: rays  (N,6)
    bboxes: bounding boxes (N,8,3)

    OUTPUT

    rgbs: color of each ray (N,3) 
    depths:  depth of each ray (N,1) 

    '''
    def forward(self, rays, bboxes, only_coarse = False,near_far=None):


        ray_mask = None

        if self.sample_method == 'NEAR_FAR':
            assert near_far is not None, 'require near_far as input '
            z_vals, sampled_rays_coarse_xyz  = self.rsp_coarse.forward(rays , near_far = near_far)
            rays_t = rays
        else:
            z_vals, sampled_rays_coarse_xyz, ray_mask  = self.rsp_coarse.forward(rays, bboxes)
            z_vals = z_vals[ray_mask]
            sampled_rays_coarse_xyz = sampled_rays_coarse_xyz[ray_mask]
            rays_t = rays[ray_mask].detach()


        if rays_t.size(0) > 1:


            z_vals = z_vals.detach()
            sampled_rays_coarse_xyz = sampled_rays_coarse_xyz.detach()

            
            rgbs, density, coeff = self.spacenet(sampled_rays_coarse_xyz, rays_t)
            coarse_density = density.clone()
            
            # density[z_vals[:,:,0]<0,:] = 0.0
            color_0, depth_0, acc_map_0, weights_0 = self.volume_render(z_vals, rgbs, density)

            if not only_coarse:


                z_samples = sample_pdf(z_vals.squeeze(), weights_0.squeeze()[...,1:-1], N_samples = self.fine_ray_sample)
                z_samples = z_samples.detach()   # (N,L)

                z_vals_fine, _ = torch.sort(torch.cat([z_vals.squeeze(), z_samples], -1), -1) #(N, L1+L2)
                samples_fine_xyz = z_vals_fine.unsqueeze(-1)*rays_t[:,:3].unsqueeze(1) + rays_t[:,3:].unsqueeze(1)  # (N,L1+L2,3)
                
                samples_fine_xyz_debug = z_samples.unsqueeze(-1)*rays_t[:,:3].unsqueeze(1) + rays_t[:,3:].unsqueeze(1)
                
                #beg = time.time()
                rgbs, density,coeff = self.spacenet_fine(samples_fine_xyz, rays_t)
                
                
                
                rgbs_debug, density_debug,coeff_debug = self.spacenet_fine(samples_fine_xyz_debug, rays_t)
                fine_density = density_debug.clone()
                
                color, depth, acc_map, weights = self.volume_render(z_vals_fine.unsqueeze(-1), rgbs, density)


                if not self.sample_method == 'NEAR_FAR':
                    color_final_0 = torch.zeros(rays.size(0),3,device = rays.device)
                    color_final_0[ray_mask] = color_0
                    depth_final_0 = torch.zeros(rays.size(0),1,device = rays.device)
                    depth_final_0[ray_mask] = depth_0
                    acc_map_final_0 = torch.zeros(rays.size(0),1,device = rays.device)
                    acc_map_final_0[ray_mask] = acc_map_0
                else:
                    color_final_0, depth_final_0, acc_map_final_0 = color_0, depth_0, acc_map_0



            else:
                if not self.sample_method == 'NEAR_FAR':
                    color_final_0 = torch.zeros(rays.size(0),3,device = rays.device)
                    color_final_0[ray_mask] = color_0
                    depth_final_0 = torch.zeros(rays.size(0),1,device = rays.device)
                    depth_final_0[ray_mask] = depth_0
                    acc_map_final_0 = torch.zeros(rays.size(0),1,device = rays.device)
                    acc_map_final_0[ray_mask] = acc_map_0
                else:
                    color_final_0, depth_final_0, acc_map_final_0 = color_0, depth_0, acc_map_0
                color, depth, acc_map, weights = color_0, depth_0, acc_map_0, weights_0
            #torch.cuda.synchronize()
            #print('render fine:',time.time()-beg)
            #print('depth range:',depth.max(),depth.min())
            #print('color range:',color.max(),color.min())
            color_final , depth_final, acc_map_final = color, depth, acc_map


            if not self.sample_method == 'NEAR_FAR':
                color_final = torch.zeros(rays.size(0),3,device = rays.device)
                color_final[ray_mask] = color
                depth_final = torch.zeros(rays.size(0),1,device = rays.device)
                depth_final[ray_mask] = depth
                acc_map_final = torch.zeros(rays.size(0),1,device = rays.device)
                acc_map_final[ray_mask] = acc_map
        else:
            color_final_0 = torch.zeros(rays.size(0),3,device = rays.device).requires_grad_()
            depth_final_0 = torch.zeros(rays.size(0),1,device = rays.device).requires_grad_()
            acc_map_final_0 = torch.zeros(rays.size(0),1,device = rays.device).requires_grad_()
            color_final, depth_final, acc_map_final = color_final_0, depth_final_0, acc_map_final_0


        return (color_final, depth_final, acc_map_final) , (color_final_0, depth_final_0, acc_map_final_0), ray_mask

    def set_max_min(self, maxs, mins):
        self.maxs = maxs
        self.mins = mins
