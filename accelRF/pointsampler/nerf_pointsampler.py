from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor
from .utils import sample_pdf

def get_init_z_vals(
    near: float, far: float, N_samples: int, 
    lindisp: bool=False, device: torch.device='cpu'):
    t_vals = torch.linspace(0., 1., steps=N_samples, device=device)
    if not lindisp:
        init_z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        init_z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))
    return init_z_vals

@torch.jit.script
def coarse_sample(
    N_samples: int,
    near: float, far: float,
    rays_o: Tensor, rays_d: Tensor,
    perturb: float=0.,
    lindisp: bool=False,
    init_z_vals: Optional[Tensor]=None,
    ):
    '''
    Args:
        N_samples: int, number of points sampled per ray
        near, far: float, from camera model
        rays_o: Tensor, the orgin of rays. [N_rays, 3]
        rays_d: Tensor, the direction of rays. [N_rays, 3]
    '''
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
        z_vals = lower + (upper - lower) * t_rand # [N_rays, N_samples]
    else:
        z_vals = init_z_vals[None, :] # [1, N_samples]
    
    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,None] # [N_rays, N_samples, 3]

    # TODO maybe u can consider adding bounding box checking here for bounded scenes.

    return pts, z_vals


@torch.jit.script
# @torch.no_grad()
def fine_sample(
    N_importance: int,
    rays_o: Tensor, rays_d: Tensor,
    z_vals: Tensor, weights: Tensor,
    det: bool=True
    ):
    '''
    fine_sample relies on (i) coarse_sample's results (ii) output of coarse MLP
    Args:
        rays_o: Tensor, the orgin of rays. [N_rays, 3]
        rays_d: Tensor, the direction of rays. [N_rays, 3]
        z_vals: Tensor, samples positional parameter in coarse sample. [N_rays|1, N_samples]
        weights: Tensor, processed weights from MLP and vol rendering. [N_rays, N_samples]
    '''
    N_rays = weights.shape[0]
    N_coarse_samples = z_vals.shape[1]
    z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1]) 

    # z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=det)
    # unfold `sample_pdf` for more fusion opportunities..
    
    # Get pdf
    device = weights.device
    weights_ = weights[..., 1:-1] + 1e-5 # prevent nans
    pdf = weights_ / torch.sum(weights_, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_importance, device=device)
        u = u.expand(N_rays, N_importance)
    else:
        u = torch.rand(N_rays, N_importance, device=device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = (inds-1).clamp_min(0)
    above = inds.clamp_max(cdf.shape[-1]-1)
    inds_g = torch.stack([below, above], -1)  # (batch, N_importance, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(z_vals_mid.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom[denom < 1e-5] = 1
    t = (u-cdf_g[...,0])/denom
    z_samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])
    
    z_vals, _ = torch.sort(torch.cat([z_vals.expand(N_rays, N_coarse_samples), z_samples], -1), -1)
    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

    return pts, z_vals

# wrap the sample functions into a `nn.Module` for better extensibility
class NeRFPointSampler(nn.Module):
    def __init__(self, N_samples: int, near: float, far: float, N_importance: int=0,
            perturb: float=0., lindisp: bool=False) -> None:
        super().__init__()
        self.N_samples = N_samples
        self.N_importance = N_importance
        self.near = near
        self.far = far
        self.perturb = perturb
        self.det = (perturb==0.)
        self.lindisp = lindisp
        self.init_z_vals = get_init_z_vals(near, far, N_samples, lindisp)
        self.init_z_vals = nn.parameter.Parameter(self.init_z_vals, requires_grad=False)
        
    @torch.no_grad()
    def forward(self, rays_o: Tensor, rays_d: Tensor, 
            z_vals: Optional[Tensor]=None, weights: Optional[Tensor]=None):

        if weights is None:
            pts, z_vals = coarse_sample(self.N_samples, self.near, self.far, 
                            rays_o, rays_d, self.perturb, init_z_vals=self.init_z_vals)
        else:
            pts, z_vals = fine_sample(self.N_importance, 
                            rays_o, rays_d, z_vals, weights, det=self.det)
        return pts, z_vals
        

