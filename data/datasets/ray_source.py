import torch
import cv2
import numpy as np
import os
from .utils import campose_to_extrinsic, read_intrinsics
from PIL import Image
import torchvision
import torch.distributions as tdist

from .ibr_dynamic import IBRDynamicDataset
from utils import ray_sampling



class IBRay_NHR(torch.utils.data.Dataset):


    def __init__(self,data_folder_path,  transforms, bunch = 4096):
        super(IBRay_NHR, self).__init__()

        self.bunch = bunch

        self.NHR_dataset = IBRDynamicDataset(data_folder_path, 1, True, transforms, [1.0, 6.5, 0.8], skip_step = 1, random_noisy = 0, holes='None')

        self.rays = []
        self.rgbs = []





        for i in range(len(self.NHR_dataset)):
            img, self.vs, _, T, K, _,_ = self.NHR_dataset.__getitem__(i)
            rays,rgbs = ray_sampling(K.unsqueeze(0), T.unsqueeze(0), (img.size(1),img.size(2)), masks = img[3,:,:].unsqueeze(0), mask_threshold = 0.5, images = img[0:3,:,:].unsqueeze(0) )
            self.rays.append(rays)
            self.rgbs.append(rgbs)
            print(i,'| generate %d rays.'%rays.size(0))
            if i>10:
                break

        self.rays = torch.cat(self.rays, dim=0)
        self.rgbs = torch.cat(self.rgbs,dim = 0)






    def __len__(self):
        return self.rays.size(0)//self.bunch

    def __getitem__(self, index):

        indexs = np.random.choice(self.rays.size(0), size=self.bunch)

        return self.rays[indexs,:], self.rgbs[indexs,:]

