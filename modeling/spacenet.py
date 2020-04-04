
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn

from utils import Trigonometric_kernel




class SpaceNet(nn.Module):


    def __init__(self, c_pos=3):
        super(SpaceNet, self).__init__()


        self.tri_kernel_pos = Trigonometric_kernel(L=10)
        self.tri_kernel_dir = Trigonometric_kernel(L=4)

        self.c_pos = c_pos

        self.pos_dim = self.tri_kernel_pos.calc_dim(c_pos)
        self.dir_dim = self.tri_kernel_dir.calc_dim(3)


        self.stage1 = nn.Sequential(
                    nn.Linear(self.pos_dim, 256),
                    nn.ReLU(inplace=True),
                    nn.Linear(256,256),
                    nn.ReLU(inplace=True),
                    nn.Linear(256,256),
                    nn.ReLU(inplace=True),
                    nn.Linear(256,256),
                    nn.ReLU(inplace=True),
                )

        self.stage2 = nn.Sequential(
                    nn.Linear(256+self.pos_dim, 256),
                    nn.ReLU(inplace=True),
                    nn.Linear(256,256),
                    nn.ReLU(inplace=True),
                    nn.Linear(256,256),
                )

        self.density_net = nn.Sequential(
                    nn.ReLU(inplace=True),
                    nn.Linear(256, 128),
                    nn.ReLU(inplace=True),
                    nn.Linear(128,1),
                    nn.ReLU(inplace=True),
                )
        self.rgb_net = nn.Sequential(
                    nn.ReLU(inplace=True),
                    nn.Linear(256+self.dir_dim, 128),
                    nn.ReLU(inplace=True),
                    nn.Linear(128,128),
                    nn.ReLU(inplace=True),
                    nn.Linear(128,128),
                    nn.ReLU(inplace=True),
                    nn.Linear(128,128),
                    nn.ReLU(inplace=True),
                    nn.Linear(128,3),
                    nn.Sigmoid()
                )

    '''
    INPUT
    pos: 3D positions (N,L,c_pos) or (N,c_pos)
    rays: corresponding rays  (N,6)

    OUTPUT

    rgb: color (N,L,3) or (N,3)
    density: (N,L,1) or (N,1)

    '''
    def forward(self, pos, rays):

        dirs = rays[...,0:3]
        
        bins_mode = False
        if len(pos)>2:
            bins_mode = True
            L = pos.size(1)
            dirs = dirs.unsqueeze(1).repeat(1,L,1)

            pos = pos.reshape((-1,self.c_pos))     #(N,c_pos)
            dirs = dirs.reshape((-1,self.c_pos))   #(N,3)


        pos = self.tri_kernel_pos(pos)
        dirs = self.tri_kernel_dir(dirs)


        x = self.stage1(pos)
        x = self.stage2(torch.cat([x,pos],dim =1))


        density = self.density_net(x)
        rgbs = self.rgb_net(torch.cat([x,dirs],dim =1))

        if bins_mode:
            density = density.reshape((-1,L,1))
            rgbs = rgbs.reshape((-1,L,3))



        return rgbs, density



         


        


        

        

