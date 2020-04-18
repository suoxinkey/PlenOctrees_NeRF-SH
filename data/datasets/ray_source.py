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


    def __init__(self,data_folder_path,  transforms, bunch = 1024,use_mask = False):
        super(IBRay_NHR, self).__init__()

        self.bunch = bunch

        self.NHR_dataset = IBRDynamicDataset(data_folder_path, 1, use_mask, transforms, [1.0, 6.5, 0.8], skip_step = 1, random_noisy = 0, holes='None')

        self.rays = []
        self.rgbs = []
        self.near_fars = []
        self.use_mask = use_mask

        if not os.path.exists(os.path.join(data_folder_path,'rays_tmp')):
            os.mkdir(os.path.join(data_folder_path,'rays_tmp'))

        if not os.path.exists(os.path.join(os.path.join(data_folder_path,'rays_tmp'),'rays_0.pt')):
            for i in range(len(self.NHR_dataset)):
                img, self.vs, frame_id, T, K, near_far,_ = self.NHR_dataset.__getitem__(i)
                img_rgb = img[0:3,:,:]
                if self.use_mask:
                    mask = img[4,:,:] *img[3,:,:] 
                    img_rgb[:, mask<0.5] = 1.0

                rays,rgbs = ray_sampling(K.unsqueeze(0), T.unsqueeze(0), (img.size(1),img.size(2)), masks = img[4,:,:].unsqueeze(0), mask_threshold = 0.5, images = img_rgb.unsqueeze(0) )
                self.rays.append(rays)
                self.rgbs.append(rgbs)
                self.near_fars.append(near_far.repeat(rays.size(0),1))   # (1,2)
                print(i,'| generate %d rays.'%rays.size(0))




            self.rays = torch.cat(self.rays, dim=0)
            self.rgbs = torch.cat(self.rgbs, dim=0)
            self.near_fars = torch.cat(self.near_fars, dim=0)   #(M,2)


            torch.save(self.rays, os.path.join(os.path.join(data_folder_path,'rays_tmp'),'rays_0.pt'))
            torch.save(self.rgbs, os.path.join(os.path.join(data_folder_path,'rays_tmp'),'rgb_0.pt'))
            torch.save(self.near_fars, os.path.join(os.path.join(data_folder_path,'rays_tmp'),'near_fars_0.pt'))
        else:
            self.rays = torch.load(os.path.join(os.path.join(data_folder_path,'rays_tmp'),'rays_0.pt'))
            self.rgbs = torch.load(os.path.join(os.path.join(data_folder_path,'rays_tmp'),'rgb_0.pt'))
            self.near_fars =  torch.load(os.path.join(os.path.join(data_folder_path,'rays_tmp'),'near_fars_0.pt'))
            img, self.vs, _, T, K, _,_ = self.NHR_dataset.__getitem__(0)
            print('load %d rays.'%self.rays.size(0))


        max_xyz = torch.max(self.vs, dim=0)[0]+0.5
        min_xyz = torch.min(self.vs, dim=0)[0]-0.5

        minx, miny, minz = min_xyz[0],min_xyz[1],min_xyz[2]
        maxx, maxy, maxz = max_xyz[0],max_xyz[1],max_xyz[2]
        bbox = np.array([[minx,miny,minz],[maxx,miny,minz],[maxx,maxy,minz],[minx,maxy,minz],[minx,miny,maxz],[maxx,miny,maxz],[maxx,maxy,maxz],[minx,maxy,maxz]])
        self.bbox = torch.from_numpy(bbox).reshape((1, 8, 3))
        







    def __len__(self):
        return self.rays.size(0)//self.bunch

    def __getitem__(self, index):

        indexs = np.random.choice(self.rays.size(0), size=self.bunch)

        return self.rays[indexs,:], self.rgbs[indexs,:], self.bbox.repeat(self.bunch,1,1), self.near_fars[indexs,:]



class IBRay_NHR_View(torch.utils.data.Dataset):


    def __init__(self,data_folder_path,  transforms, use_mask = False):
        super(IBRay_NHR_View, self).__init__()

        self.use_mask = use_mask
        self.NHR_dataset = IBRDynamicDataset(data_folder_path, 1, use_mask, transforms, [1.0, 6.5, 0.8], skip_step = 1, random_noisy = 0, holes='None')

        
        img, self.vs, _, T, K, _,_ = self.NHR_dataset.__getitem__(0)


        max_xyz = torch.max(self.vs, dim=0)[0]
        min_xyz = torch.min(self.vs, dim=0)[0]

        minx, miny, minz = min_xyz[0],min_xyz[1],min_xyz[2]
        maxx, maxy, maxz = max_xyz[0],max_xyz[1],max_xyz[2]
        bbox = np.array([[minx,miny,minz],[maxx,miny,minz],[maxx,maxy,minz],[minx,maxy,minz],[minx,miny,maxz],[maxx,miny,maxz],[maxx,maxy,maxz],[minx,maxy,maxz]])
        self.bbox = torch.from_numpy(bbox).reshape((1, 8, 3))
        

    def __len__(self):
        return 1

    def __getitem__(self, index):

        index = np.random.randint(0,len(self.NHR_dataset))

        img, self.vs, _, T, K, near_far,_ = self.NHR_dataset.__getitem__(index)

        img_rgb = img[0:3,:,:]
        if self.use_mask:
            mask = img[4,:,:] *img[3,:,:] 
            img_rgb[:, mask<0.5] = 1.0


        rays,rgbs = ray_sampling(K.unsqueeze(0), T.unsqueeze(0), (img.size(1),img.size(2)), images = img_rgb.unsqueeze(0) )

        return rays, rgbs, self.bbox.repeat(rays.size(0),1,1), img_rgb, img[3,:,:].unsqueeze(0), img[4,:,:].unsqueeze(0),near_far.repeat(rays.size(0), 1)

