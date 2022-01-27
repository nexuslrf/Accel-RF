from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor
from .._C.pointsampler import _ext

MAX_DEPTH = 10000.0

def voxel_cdf_sample(
    vox_idx: Tensor, t_near: Tensor, t_far: Tensor,
    step_size: float, fixed_samples: int=0, 
    with_base_hits: bool=True, det: bool=False):
    '''
    Based on NSVF's inverse cdf sampling
    https://github.com/facebookresearch/NSVF/blob/fairnr/modules/encoder.py#L540
    Unlike NeRF's cdf_sample, this sample keep track of intersected voxels along the ray,
    Note to avoid ambiguity, no points on the voxel boundary will be sampled (see `*.5` in CUDA kernel).

    Args:
        rays_o, rays_d: Tensor, [N_rays, 3]. 
            Due to a scalar step_size is used for all rays, rays_d here should be normalized.
        vox_idx, t_near, t_far: Tensor, [N_rays, max_hit]. ray-voxel intersection results. 
            if voxel is represented in dense voxel-grid, t_near[...,1:] == t_far[...,:-1]
        fixed_samples: int, if fixed_samples > 0, each will have the same # sampled points (fixed_samples).
    '''
    device = vox_idx.device
    N_rays = vox_idx.shape[0]
    max_hits = vox_idx.shape[-1] # the max number of hits per ray
    vox_t_range = (t_far - t_near).masked_fill(vox_idx.eq(-1), 0) 
    t_range = vox_t_range.sum(-1) # sum on n_hit per ray
    t_probs = vox_t_range / t_range[..., None]
    # get num of samples per ray.
    if fixed_samples > 0: 
        steps = torch.ones(N_rays, 1, dtype=torch.int, device=device) * fixed_samples
    else:
        steps = (t_range / step_size).ceil().int()
    
    max_steps = steps.max() 
    if with_base_hits: 
        max_steps += max_hits # include n_hits samples
    # get random noise
    noise = torch.zeros(N_rays, max_steps, device=device)
    if det:
        noise += 0.5
    else:
        noise = noise.uniform_().clamp(min=0.001, max=0.999)  # in case

    # call cuda extension
    sampled_vidx, sampled_tvals, sampled_dists = _ext.voxel_cdf_sample(
            vox_idx.contiguous(), t_near.contiguous(), t_far.contiguous(),
            noise.contiguous(), t_probs.contiguous(), steps.contiguous(), -1.
        )
    # sample_mask = sampled_vidx.ne(-1)
    # sampled_dists.clamp_min_(0.0)
    # sampled_tvals.masked_fill_(~sample_mask, MAX_DEPTH)
    # sampled_dists.masked_fill_(~sample_mask, 0.0)

    return sampled_vidx, sampled_tvals, sampled_dists

# wrap the sample functions into a `nn.Module` for better extensibility
class NSVFPointSampler(nn.Module):
    def __init__(self, step_size: float, fixed_samples: int=0, 
        with_base_hits: bool=True, det: bool=True):
        super().__init__()
        self.fixed_samples = fixed_samples
        self.with_base_hits = with_base_hits
        self.det = det 
        self.register_buffer('step_size', torch.tensor(step_size))
    
    def half_stepsize(self):
        self.step_size *= 0.5
    
    @torch.no_grad()
    def forward(self, rays_o: Tensor, rays_d: Tensor, vox_idx: Tensor, t_near: Tensor, t_far: Tensor):
        det = self.det and (not self.training)
        sampled_vidx, sampled_tvals, sampled_dists = \
            voxel_cdf_sample(vox_idx, t_near, t_far, self.step_size, 
                            self.fixed_samples, self.with_base_hits, det)
        pts = rays_o[...,None,:] + sampled_tvals[...,None] * rays_d[...,None,:] # [N_rays, max_hits, 3]
        return pts, sampled_vidx, sampled_tvals, sampled_dists
