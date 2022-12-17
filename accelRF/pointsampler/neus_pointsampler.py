import torch
import torch.nn as nn
from torch import Tensor
from .nerf_pointsampler import get_z_vals, uniform_sample, cdf_sample
from .volsdf_pointsampler import get_sphere_intersections, depth2pts_outside

@torch.jit.script                
def up_sample(z_vals: Tensor, sdf: Tensor, N_samples: int, inv_s: int):
    '''
       Args:
              z_vals: [N_rays, N_samples]
                sdf: [N_rays, N_samples]
    '''
    device = z_vals.device

    prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
    prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
    mid_sdf = (prev_sdf + next_sdf) * 0.5
    cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

    # ----------------------------------------------------------------------------------------------------------
    # Use min value of [ cos, prev_cos ]
    # Though it makes the sampling (not rendering) a little bit biased, this strategy can make the sampling more
    # robust when meeting situations like below:
    #
    # SDF
    # ^
    # |\          -----x----...
    # | \        /
    # |  x      x
    # |---\----/-------------> 0 level
    # |    \  /
    # |     \/
    # |
    # ----------------------------------------------------------------------------------------------------------
    
    prev_cos_val = torch.cat([torch.zeros([z_vals.shape[0], 1]), cos_val[:, :-1]], dim=-1)
    cos_val = torch.min(cos_val, prev_cos_val)
    cos_val = cos_val.clip(-1e3, 0.0)
    
    dist = (next_z_vals - prev_z_vals)
    prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
    next_esti_sdf = mid_sdf + cos_val * dist * 0.5
    prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
    next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
    alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
    weights = alpha * torch.cumprod(
        torch.cat([torch.ones([z_vals.shape[0], 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
    
    samples = cdf_sample(N_samples, z_vals, weights, det=True, mid_bins=False, include_init_z_vals=False)
    
    z_vals, samples_idx = torch.sort(torch.cat([z_vals, samples], -1), -1)

    return samples, z_vals, samples_idx


class NeusPointSampler(nn.Module):
    '''
    Reference: https://github.com/Totoro97/NeuS/blob/6f96f96005d72a7a358379d2b576c496a1ab68dd/models/renderer.py#L132
    '''
    def __init__(self, scene_bounding_sphere, near, far, N_samples, N_importance,
                 up_sample_steps=1, inverse_sphere_bg=False, N_samples_inverse_sphere=0,
                 with_eik_sample=False):
        super().__init__()
        self.near, self.far = near, far
        if self.far <=0:
            self.far = 2*scene_bounding_sphere
        self.N_samples = N_samples
        self.N_importance = N_importance
        self.N_samples_inverse_sphere = N_samples_inverse_sphere

        self.up_sample_steps = up_sample_steps
        self.scene_bounding_sphere = scene_bounding_sphere

        self.inverse_sphere_bg = inverse_sphere_bg
        self.with_eik_sample = with_eik_sample

    def forward(
        self, rays_o: Tensor, rays_d: Tensor, sdf_net: nn.Module, 
        density_fn: nn.Module, pts_embedder: nn.Module
    ):
        device = rays_o.device
        _ones = torch.ones(rays_o.shape[0], 1, device=device)
        near = self.near * _ones
        if not self.inverse_sphere_bg:
            far = self.far * _ones
        else:
            far = get_sphere_intersections(rays_o, rays_d, self.scene_bounding_sphere)[:,1:]

        # Start with uniform sampling
        init_z_vals = get_z_vals(near, far, self.N_samples, device=device)
        z_vals = uniform_sample(self.N_samples, rays_d.shape[0], init_z_vals=init_z_vals, 
                            perturb=1 if sdf_net.training else 0, device=rays_d.device)
        samples, samples_idx = z_vals, None

        # up sampling
        if self.N_importance > 0:
            N_steps = self.N_importance // self.up_sample_steps
            for i in range(self.up_sample_steps):
                points = rays_o.unsqueeze(1) + samples.unsqueeze(2) * rays_d.unsqueeze(1)
                points_flat = points.reshape(-1, 3)
                with torch.no_grad():
                    samples_sdf, _ = sdf_net(pts_embedder(points_flat), points_flat, sdf_only=True)

                if samples_idx is not None:
                    sdf_merge = torch.cat([sdf.reshape(-1, z_vals.shape[1] - samples.shape[1]),
                                        samples_sdf.reshape(-1, samples.shape[1])], -1)
                    sdf = torch.gather(sdf_merge, 1, samples_idx).reshape(-1, 1)
                else:
                    sdf = samples_sdf
                
                samples, z_vals, samples_idx = up_sample(z_vals, sdf, N_steps, 64 * 2**i)

        z_samples = samples

        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,None] # [N_rays, N_samples, 3]

        # add some of the near surface points
        pts_eik = None
        if self.with_eik_sample and self.training:
            idx = torch.randint(z_vals.shape[-1], (z_vals.shape[0],), device=device)
            z_samples_eik = torch.gather(z_vals, 1, idx.unsqueeze(-1))
            pts_eik = rays_o[...,None,:] + rays_d[...,None,:] * z_samples_eik[...,None]

        pts_bg, z_vals_bg = None, None
        if self.inverse_sphere_bg:
            z_vals_inverse_sphere = uniform_sample(self.N_samples_inverse_sphere, rays_d.shape[0],
                                            perturb=1 if sdf_net.training else 0, device=rays_d.device)
            z_vals_inverse_sphere = z_vals_inverse_sphere * (1./self.scene_bounding_sphere)
            z_vals_bg = torch.flip(z_vals_inverse_sphere, dims=[-1,])
            if z_vals_bg.shape[0] == 1:
                z_vals_bg = z_vals_bg.expand(rays_d.shape[0], -1)
            pts_bg = depth2pts_outside(rays_o[...,None,:], rays_d[...,None,:], z_vals_bg, 
                            self.scene_bounding_sphere)

        return pts, z_vals, pts_bg, z_vals_bg, pts_eik

    
