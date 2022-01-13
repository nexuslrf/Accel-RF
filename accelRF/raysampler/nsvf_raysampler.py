from typing import Optional
import torch
import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader
from .base import BaseRaySampler
from ..datasets import BaseDataset
from ..rep import Explicit3D

@torch.jit.script
def masked_sample(mask: Tensor, n_samples: int) -> Tensor:
    '''
    this is mainly based on NSVF's sample_pixels
    https://github.com/facebookresearch/NSVF/blob/fairnr/modules/reader.py#L122
    Most of hyperparameters are removed.
    '''
    TINY = 1e-9
    probs = mask.float() / (mask.sum() + 1e-8) # [N_rays]
    logp = torch.log(probs + TINY)
    rand_gumbel = -torch.log(-torch.log(torch.rand_like(probs) + TINY) + TINY)
    sampled_idx = (logp + rand_gumbel).topk(n_samples, dim=-1)[1]
    sampled_mask = torch.zeros_like(mask, dtype=torch.bool).scatter_(-1, sampled_idx, 1)
    return sampled_mask



class VoxIntersectRaySampler(BaseRaySampler):
    '''
    This is a higher level raysampler that uses rays info from normal raysamplers for 
    ray-voxel intersection, then filers out empty rays based intersection test results.

    **Note**: due to processing on GPU, you cannot set num_workers>0 if wrapping with dataloader.
    TODO maybe you can add a CPU version later...
    '''
    def __init__(
        self, N_rand: int,
        raysampler: BaseRaySampler, vox_rep: Explicit3D, 
        mask_sample: bool=True, device: torch.device='cuda') -> None:

        super().__init__(None, N_rand, length=raysampler.length, device=device)
        self.raysampler = raysampler
        self.vox_rep = vox_rep
        self.mask_sample = mask_sample
        self.raysampler_load = DataLoader(self.raysampler, num_workers=1, pin_memory=True)
        self.raysampler_itr = iter(self.raysampler_load)

    def __getitem__(self, index):
        ray_batch = next(self.raysampler_itr)
        rays_o, rays_d = [ray_batch[k][0].to(self.device) for k in ('rays_o', 'rays_d')] # [N_rays, 3]
        gt_rgb = ray_batch['gt_rgb'][0].to(self.device) if 'gt_rgb' in ray_batch else None

        vox_idx, t_near, t_far, hits = self.vox_rep.ray_intersect(rays_o, rays_d)
        if self.mask_sample:
            sampled_mask = masked_sample(hits, self.N_rand)
            vox_idx = vox_idx[sampled_mask]
            t_near, t_far = t_near[sampled_mask], t_far[sampled_mask]
            hits = hits[sampled_mask]
            rays_o, rays_d = rays_o[sampled_mask], rays_d[sampled_mask]
            gt_rgb = gt_rgb[sampled_mask] if gt_rgb is not None else None

        out = {
            'rays_o': rays_o, 'rays_d': rays_d, 
            'vox_idx': vox_idx, 't_near': t_near, 't_far': t_far,
            'hits': hits} # keep pay attention to hits!!!
        if gt_rgb is not None:
            out['gt_rgb'] = gt_rgb

        return out 


