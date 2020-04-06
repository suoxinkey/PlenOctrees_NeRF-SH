
import numpy as np
import torch.nn.functional as F
from torch import nn
import torch
from layers.render_layer import gen_weight

def intersection(rays, bbox):
    n = rays.shape[0]
    left_face = bbox[:, 0, 0]
    right_face = bbox[:, 6, 0]
    front_face = bbox[:, 0, 1]
    back_face = bbox[:, 6, 1]
    bottom_face = bbox[:, 0, 2]
    up_face = bbox[:, 6, 2]
    # parallel t 无穷大
    left_t = ((left_face - rays[:, 3]) / (rays[:, 0] + np.finfo(float).eps.item())).reshape((n, 1))
    right_t = ((right_face - rays[:, 3]) / (rays[:, 0] + np.finfo(float).eps.item())).reshape((n, 1))
    front_t = ((front_face - rays[:, 4]) / (rays[:, 1] + np.finfo(float).eps.item())).reshape((n, 1))
    back_t = ((back_face - rays[:, 4]) / (rays[:, 1] + np.finfo(float).eps.item())).reshape((n, 1))
    bottom_t = ((bottom_face - rays[:, 5]) / (rays[:, 2] + np.finfo(float).eps.item())).reshape((n, 1))
    up_t = ((up_face - rays[:, 5]) / (rays[:, 2] + np.finfo(float).eps)).reshape((n, 1))


    left_point = left_t * rays[:, :3] + rays[:, 3:]
    right_point = right_t * rays[:, :3] + rays[:, 3:]
    front_point = front_t * rays[:, :3] + rays[:, 3:]
    back_point = back_t * rays[:, :3] + rays[:, 3:]
    bottom_point = bottom_t * rays[:, :3] + rays[:, 3:]
    up_point = up_t * rays[:, :3] + rays[:, 3:]

    left_mask = (left_point[:, 1] >= bbox[:, 0, 1]) & (left_point[:, 1] <= bbox[:, 7, 1]) \
                & (left_point[:, 2] >= bbox[:, 0, 2]) & (left_point[:, 2] <= bbox[:, 7, 2])
    right_mask = (right_point[:, 1] >= bbox[:, 1, 1]) & (right_point[:, 1] <= bbox[:, 6, 1]) \
                 & (right_point[:, 2] >= bbox[:, 1, 2]) & (right_point[:, 2] <= bbox[:, 6, 2])

    # compare x, z
    front_mask = (front_point[:, 0] >= bbox[:, 0, 0]) & (front_point[:, 0] <= bbox[:, 5, 0]) \
                 & (front_point[:, 2] >= bbox[:, 0, 2]) & (front_point[:, 2] <= bbox[:, 5, 2])

    back_mask = (back_point[:, 0] >= bbox[:, 3, 0]) & (back_point[:, 0] <= bbox[:, 6, 0]) \
                & (back_point[:, 2] >= bbox[:, 3, 2]) & (back_point[:, 2] <= bbox[:, 6, 2])

    # compare x,y
    bottom_mask = (bottom_point[:, 0] >= bbox[:, 0, 0]) & (bottom_point[:, 0] <= bbox[:, 2, 0]) \
                  & (bottom_point[:, 1] >= bbox[:, 0, 1]) & (bottom_point[:, 1] <= bbox[:, 2, 1])

    up_mask = (up_point[:, 0] >= bbox[:, 4, 0]) & (up_point[:, 0] <= bbox[:, 6, 0]) \
              & (up_point[:, 1] >= bbox[:, 4, 1]) & (up_point[:, 1] <= bbox[:, 6, 1])

    tlist = -torch.ones_like(rays, device=rays.device)
    tlist[left_mask, 0] = left_t[left_mask].reshape((-1,))
    tlist[right_mask, 1] = right_t[right_mask].reshape((-1,))
    tlist[front_mask, 2] = front_t[front_mask].reshape((-1,))
    tlist[back_mask, 3] = back_t[back_mask].reshape((-1,))
    tlist[bottom_mask, 4] = bottom_t[bottom_mask].reshape((-1,))
    tlist[up_mask, 5] = up_t[up_mask].reshape((-1,))
    tlist = tlist.topk(k=2, dim=-1)

    return tlist[0]

class RaySamplePoint(nn.Module):
    def __init__(self, coarse_num=80):
        super(RaySamplePoint, self).__init__()
        self.coarse_num = coarse_num


    def forward(self, rays, bbox, pdf=None,  method='coarse'):
        '''
        :param rays: N*6
        :param bbox: N*8*3  0,1,2,3 bottom 4,5,6,7 up
        pdf: n*coarse_num 表示权重
        :param method:
        :return: N*C*3
        '''
        n = rays.shape[0]
        #if method=='coarse':
        sample_num = self.coarse_num
        bin_range = torch.arange(0, sample_num, device=rays.device).reshape((1, sample_num)).float()

        bin_num = sample_num
        n = rays.shape[0]
        tlist = intersection(rays, bbox)
        start = (tlist[:,1]).reshape((n,1))
        end = (tlist[:, 0]).reshape((n, 1))
        bin_sample = torch.rand((n, sample_num), device=rays.device)
        bin_width = (end - start)/bin_num
        sample_t = (bin_range + bin_sample)* bin_width + start
        sample_point = sample_t.unsqueeze(-1)*rays[:,:3].unsqueeze(1) + rays[:,3:].unsqueeze(1)
        
        return sample_t.unsqueeze(-1), sample_point


class RayDistributedSamplePoint(nn.Module):
    def __init__(self, fine_num=10):
        super(RayDistributedSamplePoint, self).__init__()
        self.fine_num = fine_num

    def forward(self, rays, depth, density, noise=0.0):
        '''
        :param rays: N*L*6
        :param depth: N*L*1
        :param density: N*L*1
        :param noise:0
        :return:
        '''

        sample_num = self.fine_num
        n = density.shape[0]

        weights = gen_weight(depth, density, noise=noise) # N*L
        weights += 1e-5
        bin = depth.squeeze()

        weights = weights[:, 1:].squeeze() #N*(L-1)
        pdf = weights/torch.sum(weights, dim=1, keepdim=True)
        cdf = torch.cumsum(pdf, dim=1)
        cdf_s = torch.cat((torch.zeros((n, 1)).type(cdf.dtype), cdf), dim=1)
        fine_bin = torch.linspace(0, 1, sample_num, device=density.device).reshape((1, sample_num)).repeat((n, 1))
        above_index = torch.ones_like(fine_bin, device=density.device).type(torch.LongTensor)
        for i in range(cdf.shape[1]):
            mask = (fine_bin > (cdf_s[:, i]).reshape((n, 1))) & (fine_bin <= (cdf[:, i]).reshape((n, 1)))
            above_index[mask] = i+1
        below_index = above_index-1
        below_index[below_index==-1]=0
        sn_below = torch.gather(bin, dim=1, index=below_index)
        sn_above = torch.gather(bin, dim=1, index=above_index)
        cdf_below = torch.gather(cdf_s, dim=1, index=below_index)
        cdf_above = torch.gather(cdf_s, dim=1, index=above_index)
        dnorm = cdf_above - cdf_below
        dnorm = torch.where(dnorm<1e-5, torch.ones_like(dnorm, device=density.device), dnorm)
        d = (fine_bin - cdf_below)/dnorm
        fine_t = (sn_above - sn_below) * d + sn_below
        fine_sample_point = fine_t.unsqueeze(-1) * rays[:, :3].unsqueeze(1) + rays[:, 3:].unsqueeze(1)
        return fine_t, fine_sample_point



