from typing import Callable, Dict, Optional, Tuple
import torch
import torch.nn as nn
from torch import Tensor
from accelRF.raysampler.utils import ndc_rays
from accelRF.render.nerf_render import volumetric_rendering

def masked_scatter(mask: torch.BoolTensor, x: Tensor):
    B, K = mask.size()
    if x.dim() == 1:
        return x.new_zeros(B, K).masked_scatter(mask, x)
    return x.new_zeros(B, K, x.size(-1)).masked_scatter(
        mask.unsqueeze(-1).expand(B, K, x.size(-1)), x)

class NSVFRender(nn.Module):
    def __init__(
        self,
        pts_embedder: Optional[nn.Module],
        view_embedder: Optional[nn.Module],
        voxel_embedder: nn.Module,
        model: nn.Module,
        point_sampler: nn.Module,
        white_bkgd: bool=False,
        chunk: int=1024*16
    ):
        super().__init__()
        self.pts_embedder = pts_embedder if pts_embedder is not None else nn.Identity()
        self.view_embedder = view_embedder if view_embedder is not None else nn.Identity()
        self.voxel_embedder = voxel_embedder
        self.model = model
        self.point_sampler = point_sampler
        self.white_bkgd = white_bkgd
        self.chunk = chunk

    def forward(self, rays_o: Tensor, rays_d: Tensor, vox_idx: Tensor, 
        t_near: Tensor, t_far: Tensor, ray_hits: Tensor):
        '''
        Args:
            rays_o, rays_d: [N_rays, 3]
            vox_idx, t_near, t_far: [N_rays, max_hits]
            hits: [N_rays]
        '''
        # TODO chunk forward
        # TODO early stopping
        # maintain the original rays
        rays_o_ = rays_o.reshape(-1, 3).clone()
        rays_d_ = rays_d.reshape(-1, 3).clone()
        
        if ray_hits.sum() > 0:
            vox_idx, t_near, t_far = vox_idx[ray_hits], t_near[ray_hits], t_far[ray_hits] # [n_hits, max_hits]
            rays_o, rays_d = rays_o[ray_hits], rays_d[ray_hits] # [n_hits, 3]
            # point sampling
            sample_out = self.point_sampler(rays_o, rays_d, vox_idx, t_near, t_far)
            pts, p2v_idx, t_vals = sample_out # [n_hits, max_samples] + [3], [], []
            
            nn_out, n_pts = self.mask_forward(pts, p2v_idx, rays_d)
            
            r_out = volumetric_rendering(
                nn_out['rgb'], nn_out['sigma'], t_vals, white_bkgd=self.white_bkgd)
        
        ret = {'rgb': r_out['rgb'], 'disp': r_out['disp'], 'acc': r_out['acc']}
        return r_out

    def mask_forward(self, pts: Tensor, p2v_idx: Tensor, rays_d: Tensor):
        mask_pts = p2v_idx.ne(-1) 
        n_pts = mask_pts.sum()
        if n_pts == 0:
            return None, 0

        rays_d = rays_d[...,None,:].expand_as(pts)[mask_pts] # [n_pts, 3]
        pts, p2v_idx = pts[mask_pts], p2v_idx[mask_pts]
        
        vox_embeds = self.voxel_embedder(pts, p2v_idx)
        nn_out = self.model(
            self.pts_embedder(vox_embeds), self.view_embedder(rays_d)
        )

        # scatter back...
        out = {
            'sigma': masked_scatter(nn_out['sigma']),
            'rgb': masked_scatter(nn_out['rgb'])
        }
        return out, n_pts