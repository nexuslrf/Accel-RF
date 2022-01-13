from typing import Tuple
import numpy as np
import torch

# Ray helpers
@torch.jit.script
def get_rays(H: int, W: int, focal: float, c2w: torch.Tensor, normalize_dir: bool=False):
    
    device = c2w.device
    i, j = torch.meshgrid(
        torch.linspace(0, W-1, W, device=device), 
        torch.linspace(0, H-1, H, device=device))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -torch.ones_like(i, device=device)], -1) # note this func is for openGL coor
    # in openCV, you may need: dirs = torch.stack([(i-W*.5)/focal, (j-H*.5)/focal, torch.ones_like(i, device=device)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    if normalize_dir:
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    return rays_o, rays_d

def get_rays_openCV():
    pass

def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    if isinstance(K, np.ndarray):
        dirs = np.stack([(i-W*.5)/K[0,0], -(j-H*.5)/K[1,1], -np.ones_like(i)], -1)
    else: # is scalar
        dirs = np.stack([(i-W*.5)/K, -(j-H*.5)/K, -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d

@torch.jit.script
def ndc_rays(H: int, W: int, focal: float, near: float, rays_o: torch.Tensor, rays_d: torch.Tensor):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d

@torch.jit.script
def aabb_intersect(ray_o: torch.Tensor, ray_d: torch.Tensor, center_pts: torch.Tensor, 
    radius: float, near: float, far: float):

    tbot = (center_pts - radius - ray_o) / ray_d
    ttop = (center_pts + radius - ray_o) / ray_d
    tmin = torch.where(tbot < ttop, tbot, ttop)
    tmax = torch.where(tbot > ttop, tbot, ttop)
    largest_tmin, _ = tmin.max(dim=1)
    smallest_tmax, _ = tmax.min(dim=1)
    t_near = largest_tmin.clamp_min(near)
    t_far = smallest_tmax.clamp_max(far)
    return t_near, t_far