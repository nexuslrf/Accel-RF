from typing import Optional
import torch
from torch import Tensor

def get_init_z_vals(
    near: float, far: float, N_samples: int, 
    lindisp: bool=False, device: torch.device='cuda'):
    t_vals = torch.linspace(0., 1., steps=N_samples, device=device)
    if not lindisp:
        init_z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        init_z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))
    return init_z_vals

@torch.jit.script
def uniform_sample(
    N_samples: int,
    near: float, far: float,
    rays_o: Tensor, rays_d: Tensor,
    perturb: float=0.,
    lindisp: bool=False,
    init_z_vals: Optional[Tensor]=None,
    ):

    device = rays_o.device
    N_rays = rays_o.shape[0]
    if init_z_vals is None:
        init_z_vals = get_init_z_vals(near, far, N_samples, lindisp, device) # (N_samples)

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (init_z_vals[1:] + init_z_vals[:-1])
        upper = torch.cat([mids, init_z_vals[-1:]], -1)
        lower = torch.cat([init_z_vals[:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand([N_rays, N_samples]) * perturb
        z_vals = lower + (upper - lower) * t_rand
    else:
        z_vals = init_z_vals[None, :]
    
    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,None] # [N_rays, N_samples, 3]

    # TODO maybe u can consider adding bounding box checking here for bounded scenes.

    return pts

    


