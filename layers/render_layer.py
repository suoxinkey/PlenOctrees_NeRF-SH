import torch
import torch.nn as nn
import torch.nn.functional as F


class VolumeRenderer(nn.Module):
    def __init__(self):
        super(VolumeRenderer, self).__init__()

        self.sigma2alpha = lambda sigma, delta, act_fn=F.relu: 1.-torch.exp(-act_fn(sigma)*delta)


    def forward(self, depth, rgb, sigma, noise=0.):
        """
        N - num rays; L - num bins; 
        :param depth: torch.tensor, depth for each sample along the ray. [N, L, 1]
        :param rgb: torch.tensor, raw rgb output from the network. [N, L, 3]
        :param sigma: torch.tensor, raw density (without activation). [N, L, 1]
        
        :return:
            color: torch.tensor [N, 3]
            depth: torch.tensor [N, 1]
        """

        delta = (depth[:, 1:] - depth[:, :-1]).squeeze()  # [N, L-1]
        pad = torch.Tensor([1e10]).expand_as(delta[...,:1]).to(delta.device)
        delta = torch.cat([delta, pad], dim=-1)   # [N, L]

        if noise > 0.:
            sigma += (torch.normal(size=sigma.size()) * noise)

        alpha = self.sigma2alpha(sigma, delta.unsqueeze(-1))
        weights = torch.mul(alpha, torch.cumprod(1.-alpha+1e-10, dim=-1))   #[N, L, 1]

        color = torch.sum(torch.mul(F.sigmoid(rgb), weights), dim=1) #[N, 3]

        depth = torch.sum(weights * depth, dim=1)   # [N, 1]
        acc = torch.sum(weights, dim=1).unsqueeze(-1)   
        
        return color, depth


if __name__ == "__main__":
    N_rays = 1024
    N_samples = 32

    depth = torch.randn(N_rays, N_samples, 1)
    raw  = torch.randn(N_rays, N_samples, 3)
    sigma = torch.randn(N_rays, N_samples, 1)

    renderer = VolumeRenderer()

    color, dpt = renderer(depth, raw, sigma)
    print('Predicted [CPU]: ', color.shape, dpt.shape)

    if torch.cuda.is_available():
        depth = depth.cuda()
        raw = raw.cuda()
        sigma = sigma.cuda()
        renderer = renderer.cuda()

        color, dpt = renderer(depth, raw, sigma)
        print('Predicted [GPU]: ', color.shape, dpt.shape)


################################## ORIGIONAL IMPLEMENTATION ##################################################
# def raw2outputs(raw, z_vals, rays_d):
#     raw2alpha = lambda raw, dists, act_fn=tf.nn.relu: 1.-tf.exp(-act_fn(raw)*dists)
    
#     dists = z_vals[...,1:] - z_vals[...,:-1]
#     dists = tf.concat([dists, tf.broadcast_to([1e10], dists[...,:1].shape)], -1) # [N_rays, N_samples]
    
#     dists = dists * tf.linalg.norm(rays_d[...,None,:], axis=-1)

#     rgb = tf.math.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
#     noise = 0.
#     if raw_noise_std > 0.:
#         noise = tf.random.normal(raw[...,3].shape) * raw_noise_std
#     alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
#     weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
#     rgb_map = tf.reduce_sum(weights[...,None] * rgb, -2)  # [N_rays, 3]
    
#     depth_map = tf.reduce_sum(weights * z_vals, -1) 
#     disp_map = 1./tf.maximum(1e-10, depth_map / tf.reduce_sum(weights, -1))
#     acc_map = tf.reduce_sum(weights, -1)
    
#     if white_bkgd:
#         rgb_map = rgb_map + (1.-acc_map[...,None])
    
#     return rgb_map, disp_map, acc_map, weights, depth_map
###############################################################################################################