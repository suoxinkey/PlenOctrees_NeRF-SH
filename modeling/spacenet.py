
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
import time

from utils import Trigonometric_kernel




class SpaceNet(nn.Module):


    def __init__(self, c_pos=3, include_input = True):
        super(SpaceNet, self).__init__()


        self.tri_kernel_pos = Trigonometric_kernel(L=10,include_input = include_input)
        self.tri_kernel_dir = Trigonometric_kernel(L=4, include_input = include_input)

        self.c_pos = c_pos

        self.pos_dim = self.tri_kernel_pos.calc_dim(c_pos)
        self.dir_dim = self.tri_kernel_dir.calc_dim(3)

        backbone_dim = 256
        head_dim = 128


        self.stage1 = nn.Sequential(
                    nn.Linear(self.pos_dim, backbone_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(backbone_dim,backbone_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(backbone_dim,backbone_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(backbone_dim,backbone_dim),
                    nn.ReLU(inplace=True),
                )

        self.stage2 = nn.Sequential(
                    nn.Linear(backbone_dim+self.pos_dim, backbone_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(backbone_dim,backbone_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(backbone_dim,backbone_dim),
                )

        self.density_net = nn.Sequential(
                    nn.ReLU(inplace=True),
                    nn.Linear(backbone_dim, head_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(head_dim,1)
                )
        self.rgb_net = nn.Sequential(
                    nn.ReLU(inplace=True),
                    nn.Linear(backbone_dim+self.dir_dim, head_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(head_dim,head_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(head_dim,head_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(head_dim,head_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(head_dim,3)
                )


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

            dirs = rays[...,0:3]
            
        bins_mode = False
        if len(pos.size())>2:
            bins_mode = True
            L = pos.size(1)
            pos = pos.reshape((-1,self.c_pos))     #(N,c_pos)
            if rays is not None:
                dirs = dirs.unsqueeze(1).repeat(1,L,1)
                dirs = dirs.reshape((-1,self.c_pos))   #(N,3)

            
           

        if maxs is not None:
            pos = ((pos - mins)/(maxs-mins) - 0.5)*2

        pos = self.tri_kernel_pos(pos)
        if rays is not None:
            dirs = self.tri_kernel_dir(dirs)

        #torch.cuda.synchronize()
        #print('transform :',time.time()-beg)

        #beg = time.time()
        x = self.stage1(pos)
        x = self.stage2(torch.cat([x,pos],dim =1))


        density = self.density_net(x)


        if rays is not None:
            rgbs = self.rgb_net(torch.cat([x,dirs],dim =1))

        #torch.cuda.synchronize()
        #print('fc:',time.time()-beg)

        if bins_mode:
            density = density.reshape((-1,L,1))
            if rays is not None:
                rgbs = rgbs.reshape((-1,L,3))





        return rgbs, density



         


        


        

        


