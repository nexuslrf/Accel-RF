from typing import Dict, Optional
import torch
from torch import Tensor
import numpy as np

# volumetric rendering should be differentialable.
@torch.jit.script
def volumetric_rendering(
    rgb: Optional[Tensor], sigma: Tensor, z_vals: Tensor, dirs: Tensor, 
    white_bkgd: bool=False) -> Dict[str, Tensor]:
    """Volumetric Rendering Function.

    Args:
        rgb: Tensor, color, [N_rays, N_samples, 3]
        sigma: Tensor, density, [N_rays, N_samples, 1].
        z_vals: Tensor, [N_rays, N_samples].
        dirs: Tensor, [N_rays, 3]. Note: *no expand*
        white_bkgd: bool.

    Returns:
        dict(
            comp_rgb: Tensor, [N_rays, 3].
            disp: Tensor, [N_rays].
            acc: Tensor, [N_rays].
            weights: Tensor, [N_rays, N_samples]
        )
    """
    eps = 1e-10
    dists = torch.cat([
        z_vals[..., 1:] - z_vals[..., :-1],
        1e10 * torch.ones_like(z_vals[..., :1])
    ], -1) # [N_rays, N_samples]

    # Multiply each distance by the norm of its corresponding direction ray
    # to convert to real world distance (accounts for non-unit directions).
    dists = dists * torch.norm(dirs[..., None, :], dim=-1)
    
    # random noise is omitted.
    # noise = torch.randn_like(sigma) * noise_std

    # Note that we're quietly turning sigma from [..., 0] to [...].
    alpha = 1.0 - torch.exp(-sigma[..., 0] * dists)
    accum_prod = torch.cat([
        torch.ones_like(alpha[..., :1]),
        torch.cumprod(1.0 - alpha[..., :-1] + eps, -1)
    ], -1)
    weights = alpha * accum_prod
    if rgb is None:
        return {'weights': weights}
    else:
        comp_rgb = (weights[..., None] * rgb).sum(-2)
        depth = (weights * z_vals).sum(-1)
        acc = weights.sum(-1)
        # Equivalent to (but slightly more efficient and stable than):
        #  disp = 1 / max(eps, where(acc > eps, depth / acc, 0))
        inv_eps = torch.tensor(1 / eps, dtype=depth.dtype, device=depth.device)
        disp = acc / depth
        disp = torch.where((disp > 0) & (disp < inv_eps) & (acc > eps), disp, inv_eps)
        if white_bkgd:
            comp_rgb = comp_rgb + (1. - acc[..., None])
    return {'rgb':comp_rgb, 'disp':disp, 'acc':acc, 'weights':weights}