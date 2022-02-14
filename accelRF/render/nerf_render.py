from typing import Callable, Dict, Optional, Tuple
import torch
import torch.nn as nn
from torch import Tensor

from accelRF.raysampler.utils import ndc_rays


# volumetric rendering should be differentialable.
@torch.jit.script
def volumetric_rendering(
    rgb: Optional[Tensor], sigma: Tensor, 
    z_vals: Tensor, dir_lens: Optional[Tensor]=None, 
    white_bkgd: bool=False, with_dist: bool=False,
    rgb_only: bool=False, with_T: bool=False) -> Dict[str, Tensor]:
    """Volumetric Rendering Function.

    Args:
        rgb: Tensor, color, [N_rays, N_samples, 3]
        sigma: Tensor, density, [N_rays, N_samples, 1].
        z_vals: Tensor, [N_rays, N_samples]. Used for calculating dist
            if is None, input sigma should already multiplied with dist.
        dir_lens: [N_rays, 1]
        white_bkgd: bool.
            if True, sigma is actually sigma * dist. 
        dirs: Tensor, [N_rays, 3]. Note: *no expand* Update: **removed**

    Returns:
        dict(
            comp_rgb: Tensor, [N_rays, 3].
            disp: Tensor, [N_rays].
            acc: Tensor, [N_rays].
            weights: Tensor, [N_rays, N_samples]
        )
    """
    eps = 1e-10
    ret = {}
    if not with_dist:
        dists = torch.cat([
            z_vals[..., 1:] - z_vals[..., :-1],
            1e10 * torch.ones_like(z_vals[..., :1])
        ], -1) # [N_rays, N_samples]
        # Multiply each distance by the norm of its corresponding direction ray
        # to convert to real world distance (accounts for non-unit directions).
        if dir_lens is not None:
            dists = dists * dir_lens
        # Note that we're quietly turning sigma from [..., 0] to [...].
        free_energy = sigma[..., 0] * dists
    else:
        free_energy = sigma[..., 0]
        
    # random noise is omitted.
    # noise = torch.randn_like(sigma) * noise_std
    
    alpha = 1.0 - torch.exp(-free_energy)
    transmittance = torch.cat([
        torch.ones_like(alpha[..., :1]), # add 0 for transperancy 1 at t_0
        torch.cumprod(1.0 - alpha + eps, -1)
    ], -1)
    weights = alpha * transmittance[..., :-1]
    ret['weights'] = weights
    if with_T: 
        ret['transmittance'] = transmittance
    if rgb is not None:
        comp_rgb = (weights[..., None] * rgb).sum(-2)
        acc = weights.sum(-1)
        if white_bkgd:
            comp_rgb = comp_rgb + (1. - acc[..., None])
        ret['rgb'] = comp_rgb
        if not rgb_only:
            depth = (weights * z_vals).sum(-1)
            # Equivalent to (but slightly more efficient and stable than):
            #  disp = 1 / max(eps, where(acc > eps, depth / acc, 0))
            inv_eps = torch.tensor(1 / eps, dtype=depth.dtype, device=depth.device)
            disp = acc / depth
            disp = torch.where((disp > 0) & (disp < inv_eps) & (acc > eps), disp, inv_eps)
            ret['disp'], ret['acc'] = disp, acc
    return ret


class NeRFRender(nn.Module):
    '''
    Wrap up core components in one nn.Module to show a clear workflow and 
    can better utilize multi-GPU training.
    TODO starts with `NeRFRender` class, if this function generalize well, then do a higher level abstraction.
    '''
    def __init__(
        self,
        embedder_pts: Optional[nn.Module],
        embedder_views: Optional[nn.Module],
        model: nn.Module,
        point_sampler: nn.Module,
        fine_model: Optional[nn.Module]=None,
        white_bkgd: bool=False,
        fast_eval: bool=True,
        use_ndc: bool=False,
        hwf: Optional[Tuple]=None,
        chunk: int=1024*16
        ):
        super().__init__()
        self.embedder_pts = embedder_pts if embedder_pts is not None else nn.Identity()
        self.embedder_views = embedder_views if embedder_views is not None else nn.Identity()
        self.model = model
        self.hierachical = False
        if fine_model is not None:
            self.fine_model = fine_model
            self.hierachical = True
        self.point_sampler = point_sampler
        self.white_bkgd = white_bkgd
        self.fast_eval = fast_eval
        self.chunk = chunk
        self.use_ndc = use_ndc
        self.hwf = hwf
        if use_ndc: assert hwf is not None, "NDC need hwf."

    def jit_script(self):
        self.embedder_pts = torch.jit.script(self.embedder_pts)
        self.embedder_views = torch.jit.script(self.embedder_views)
        self.model = torch.jit.script(self.model)
        self.fine_model = torch.jit.script(self.fine_model) if self.hierachical else None
        return self


    def forward(self, rays_o: Tensor, rays_d: Tensor):
        '''
        Args:
            rays_o: Tensor, sampled ray origins, [N_rays, 3]
            rays_d: Tensor, sampled ray directions, [N_rays, 3]
        '''
        N_rays = rays_o.shape[0]
        if N_rays > self.chunk:
            ret = [self.forward(rays_o[ci:ci+self.chunk], rays_d[ci:ci+self.chunk]) for ci in range(0, N_rays, self.chunk)]
            ret = {
                k: torch.cat([out[k] for out in ret], 0)
                for k in ret[0]
            }
            return ret
        else:
            ret = {}
            skip_coarse_rgb = (not self.training and self.fast_eval and not self.hierachical)

            dir_lens = torch.norm(rays_d, dim=-1, keepdim=True) # [N_rays, 1]
            viewdirs = (rays_d / dir_lens)[...,None,:] # normalize the view direction, [N_rays, 1, 3]
            if self.use_ndc:
                rays_o, rays_d = ndc_rays(*self.hwf, 1., rays_o, rays_d)
            # start point sampling
            sample_out = self.point_sampler(rays_o, rays_d)
            pts, z_vals = sample_out # [N_rays, N_samples, 3], [N_rays|1, N_samples]
            
            pts_embed = self.embedder_pts(pts) # [N_rays, N_samples, pe_dim]
            view_embed = self.embedder_views(viewdirs) # [N_rays, 1, ve_dim]
            
            if not skip_coarse_rgb:
                nn_out = self.model(pts_embed, view_embed.expand(*pts.shape[:-1],-1)) # rgb: [N_rays, N_samples, 3], sigma: [N_rays, N_samples, 1]  
                r_out = volumetric_rendering(nn_out['rgb'], nn_out['sigma'], z_vals, dir_lens, self.white_bkgd)
            else:
                nn_out = self.model(pts_embed) # only sigma: [N_rays, N_samples, 1]  
                r_out = volumetric_rendering(None, nn_out['sigma'], z_vals, dir_lens, self.white_bkgd) # only weights
            
            if self.hierachical:
                if not skip_coarse_rgb:
                    ret = {'rgb0': r_out['rgb'], 'disp0': r_out['disp'], 'acc0': r_out['acc']}
                sample_out = self.point_sampler(rays_o, rays_d, z_vals, r_out['weights'])
                pts, z_vals = sample_out # # [N_rays, N_samples_f, 3], [N_rays|1, N_samples_f]
                pts_embed = self.embedder_pts(pts) # [N_rays, N_samples_f, pe_dim]
                # reuse `view_embed`
                nn_out = self.fine_model(pts_embed, view_embed.expand(*pts.shape[:-1],-1))
                r_out = volumetric_rendering(nn_out['rgb'], nn_out['sigma'], z_vals, dir_lens, self.white_bkgd)
            
            ret = {'rgb': r_out['rgb'], 'disp': r_out['disp'], 'acc': r_out['acc'], **ret}

            return ret


