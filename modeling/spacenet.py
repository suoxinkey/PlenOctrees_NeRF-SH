
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
import time

from utils import Trigonometric_kernel, computeRGB


class DenseLayer(nn.Linear):
    def __init__(self, in_dim: int, out_dim: int, activation: str = "relu", *args, **kwargs) -> None:
        self.activation = activation
        super().__init__(in_dim, out_dim, *args, **kwargs)

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(self.weight, gain=torch.nn.init.calculate_gain(self.activation))
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

class SpaceNet(nn.Module):

    def __init__(self, c_pos=3, include_input = True, use_sh = True):
        super(SpaceNet, self).__init__()


        self.tri_kernel_pos = Trigonometric_kernel(L=10,include_input = include_input)
        self.tri_kernel_dir = Trigonometric_kernel(L=4, include_input = include_input)

        self.c_pos = c_pos

        self.pos_dim = self.tri_kernel_pos.calc_dim(c_pos)
        self.dir_dim = self.tri_kernel_dir.calc_dim(3)

        self.use_sh = use_sh

        backbone_dim = 256
        head_dim = 128
        W = backbone_dim
        D = 8
        input_ch = self.pos_dim
        self.skips = [4]
        self.pts_linears = nn.ModuleList(
            [DenseLayer(input_ch, W, activation="relu")] + [DenseLayer(W, W, activation="relu") if i not in self.skips else DenseLayer(W + input_ch, W, activation="relu") for i in range(D-1)])
       

        if(use_sh):
            output_ch = 28
            self.output_linear = DenseLayer(W, output_ch, activation="linear")
        else:
            self.feature_linear = DenseLayer(W, W, activation="linear")
            self.alpha_linear = DenseLayer(W, 1, activation="linear")
            self.rgb_linear = DenseLayer(W//2, 3, activation="linear")

        # self.stage1 = nn.Sequential(
        #             nn.Linear(self.pos_dim, backbone_dim),
        #             nn.ReLU(inplace=True),
        #             nn.Linear(backbone_dim,backbone_dim),
        #             nn.ReLU(inplace=True),
        #             nn.Linear(backbone_dim,backbone_dim),
        #             nn.ReLU(inplace=True),
        #             nn.Linear(backbone_dim,backbone_dim),
        #             nn.ReLU(inplace=True),
        #         )

        # self.stage2 = nn.Sequential(
        #             nn.Linear(backbone_dim+self.pos_dim, backbone_dim),
        #             nn.ReLU(inplace=True),
        #             nn.Linear(backbone_dim,backbone_dim),
        #             nn.ReLU(inplace=True),
        #             nn.Linear(backbone_dim,backbone_dim),
        #         )

        # self.density_net = nn.Sequential(
        #             nn.ReLU(inplace=False),
        #             nn.Linear(backbone_dim, 1)
        #         )
        # self.rgb_net = nn.Sequential(
        #             nn.ReLU(inplace=False),
        #             nn.Linear(backbone_dim, head_dim),
        #             nn.ReLU(inplace=True),
        #             nn.Linear(head_dim,rgb_dim)
        #         )


    '''
    INPUT
    pos: 3D positions (N,L,c_pos) or (N,c_pos)
    rays: corresponding rays  (N,6)

    OUTPUT

    rgb: color (N,L,3) or (N,3)
    density: (N,L,1) or (N,1)

    '''
    def forward(self, pos, rays, maxs=None, mins=None):


        #beg = time.time()
        rgbs = None
        if rays is not None:

            view_dirs = rays[...,0:3]
            
        bins_mode = False
        if len(pos.size())>2:
            bins_mode = True
            L = pos.size(1)
            pos = pos.reshape((-1,self.c_pos))     #(N,c_pos)
            if rays is not None:
                view_dirs = view_dirs.unsqueeze(1).repeat(1,L,1)
                view_dirs = view_dirs.reshape((-1,self.c_pos))   #(N,3)

        pos_kernel = self.tri_kernel_pos(pos)
        if rays is not None:
            dirs_kernel = self.tri_kernel_dir(view_dirs)

        # x = self.stage1(pos)
        # x = self.stage2(torch.cat([x,pos],dim =1))
        # density = self.density_net(x)
        h = pos_kernel
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([pos_kernel, h], -1)
        
        if not self.use_sh:
            density = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, dirs_kernel], -1)
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)
            rgbs = self.rgb_linear(h)
            
        else:
            outputs = self.output_linear(h)
            density = outputs[:, :1].reshape((-1, 1))
            coeff = outputs[:, 1:].reshape((-1, 9, 3))
            if(rays is not None):
                rgbs = computeRGB(view_dirs, coeff)
            else:
                rgbs = torch.zeros((outputs.shape[0], 3)).cuda()

        if bins_mode:
            density = density.reshape((-1,L,1))
            if rays is not None:
                rgbs = rgbs.reshape((-1,L,3))

        return rgbs, density, outputs



         


        


        

        


