from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor

def get_z_vals(
    near: float, far: float, N_samples: int, 
    lindisp: bool=False, device: torch.device='cpu'):
    t_vals = torch.linspace(0., 1., steps=N_samples, device=device)[None,:]
    if not lindisp:
        init_z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        init_z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))
    return init_z_vals

@torch.jit.script
def uniform_sample(
    N_samples: int,
    rays_o: Tensor, rays_d: Tensor,
    near: float=0., far: float=1.,
    perturb: float=0.,
    lindisp: bool=False,
    init_z_vals: Optional[Tensor]=None, # [1, N_samples]
    only_z_vals: bool=False,
    ):
    '''
    Args:
        N_samples: int, number of points sampled per ray
        near, far: float, from camera model
        rays_o: Tensor, the orgin of rays. [N_rays, 3]
        rays_d: Tensor, the direction of rays. [N_rays, 3]
    Return:
        pts: Tensor, sampled points. [N_rays, N_samples, 3]
        z_vals: Tensor, [N_rays, N_samples] or [1, N_samples]
    '''
    device = rays_o.device
    N_rays = rays_o.shape[0]
    if init_z_vals is None:
        init_z_vals = get_z_vals(near, far, N_samples, lindisp, device) # (1, N_samples)

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (init_z_vals[1:] + init_z_vals[:-1])
        upper = torch.cat([mids, init_z_vals[-1:]], -1)
        lower = torch.cat([init_z_vals[:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand([N_rays, N_samples], device=device) * perturb
        z_vals = lower + (upper - lower) * t_rand # [N_rays, N_samples]
    else:
        z_vals = init_z_vals # [1, N_samples]
    
    if only_z_vals:
        return None, z_vals
    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,None] # [N_rays, N_samples, 3]

    # TODO maybe u can consider adding bounding box checking here for bounded scenes.

    return pts, z_vals


@torch.jit.script
# @torch.no_grad()
def cdf_sample(
    N_samples: int,
    rays_o: Tensor, rays_d: Tensor,
    z_vals: Tensor, weights: Tensor,
    det: bool=True
    ):
    '''
    cdf_sample relies on (i) coarse_sample's results (ii) output of coarse MLP
    In this function, each ray will have the same number of sampled points, 
    there's  voxel_cdf_sample function, that sample variable points for different rays.
    TODO@chensjtu we also plan to write a CUDA version of normal cdf sample, which can avoid using sort.
    Args:
        rays_o: Tensor, the orgin of rays. [N_rays, 3]
        rays_d: Tensor, the direction of rays. [N_rays, 3]
        z_vals: Tensor, samples positional parameter in coarse sample. [N_rays|1, N_samples]
        weights: Tensor, processed weights from MLP and vol rendering. [N_rays, N_samples]
    '''
    N_rays = weights.shape[0]
    N_base_samples = z_vals.shape[1]
    z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1]) 

    # z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_samples, det=det)
    # unfold `sample_pdf` for more fusion opportunities..
    
    # Get pdf & cdf
    device = weights.device
    weights_ = weights[..., 1:-1] + 1e-5 # prevent nans
    pdf = weights_ / torch.sum(weights_, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples, device=device)
        u = u.expand(N_rays, N_samples)
    else:
        u = torch.rand(N_rays, N_samples, device=device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = (inds-1).clamp_min(0)
    above = inds.clamp_max(cdf.shape[-1]-1)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(z_vals_mid.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom[denom < 1e-5] = 1
    t = (u-cdf_g[...,0])/denom
    z_samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])
    
    z_vals, _ = torch.sort(torch.cat([z_vals.expand(N_rays, N_base_samples), z_samples], -1), -1)
    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_samples, 3]

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
        init_z_vals = get_z_vals(near, far, N_samples, lindisp)
        self.register_buffer('init_z_vals', init_z_vals, False) # no need to checkpoint
        
    @torch.no_grad()
    def forward(self, rays_o: Tensor, rays_d: Tensor, 
            z_vals: Optional[Tensor]=None, weights: Optional[Tensor]=None):

        if weights is None:
            pts, z_vals = uniform_sample(self.N_samples, rays_o, rays_d, 
                            self.near, self.far, self.perturb, init_z_vals=self.init_z_vals)
        else:
            pts, z_vals = cdf_sample(self.N_importance, 
                            rays_o, rays_d, z_vals, weights, det=self.det)
        return pts, z_vals
        

