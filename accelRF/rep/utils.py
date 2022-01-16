from typing import Optional
from numpy import interp
import torch
from torch import Tensor

@torch.jit.script
def bbox2voxels(bbox: Tensor, voxel_size: float) -> Tensor:
    '''
    bbox2voxel: https://github.com/facebookresearch/NSVF/fairnr/modules/encoder.py#L1053
    bbox: array [min_x,y,z, max_x,y,z]
    '''
    v_min, v_max = bbox[:3], bbox[3:]
    steps = ((v_max - v_min) / voxel_size).round().long() + 1
    # note the difference between torch.meshgrid and np.meshgrid. torch.meshgrid is better.
    points = torch.stack(torch.meshgrid([torch.arange(s) for s in steps]), -1) # s_x,s_y,s_z,3
    points = points * voxel_size + v_min
    return points

@torch.jit.script
def trilinear_interp(sample_pts: Tensor, center_pts: Tensor, corner_feats: Tensor,
    voxel_size: float, offset: Optional[Tensor]=None) -> Tensor:
    '''
    Args:
        sample_pts, center_pts: [N, 3]
        corner_feats: [N, 8, embed_dim]
        offset: [8, 3]
    '''
    if offset is None:
        offset = torch.stack(
                torch.meshgrid([torch.tensor([0.,1.], device=sample_pts.device)]*3),-1
            ).reshape(-1,3)
    p = ((sample_pts - center_pts) / voxel_size + 0.5)[...,None,:] # +0.5 to rescale to [0,1], [N, 1, 3]
    r = (2*p*offset - p - offset + 1).prod(dim=-1, keepdim=True) # <=> (p*offset + (1-p)*(1-offset)); [N, 8, 1]
    interp_feat = (corner_feats * r).sum(1) # [N, embed_dim]
    return interp_feat

def offset_points(bits: int=2, n_dim: int=3, 
    scale :float=0, device :torch.device='cpu') -> Tensor:
    '''
    Return bits^n_dim meshgrid cube.
    Args:
        scale: -1 or 0. -1 -> [-1,1]; 0 -> [0, 1]
    '''
    c = torch.arange(bits, device=device)
    offset = torch.stack(torch.meshgrid([c]*n_dim), -1).reshape(-1, n_dim) / (bits - 1.)
    if scale == -1:
        offset = 2 * offset - 1
    return offset

def discretize_points(pts, voxel_size):
    min_pts = pts.min(0, keepdim=True)[0]
    coords = ((pts - min_pts) / voxel_size).round_().long()
    return coords

def get_corner_mapping(center_pts: Tensor, voxel_size: float):
    center_coords = discretize_points(center_pts, voxel_size) # [N ,3]
    offset = offset_points(device=center_pts.device) # [8,3], scale [0,1]
    corner_coords = (center_coords[:,None,:] + offset.long()).reshape(-1, 3) # [N*8, 3]
    unique_coords, center2corner = torch.unique(corner_coords, dim=0, sorted=True, return_inverse=True)
    center2corner = center2corner.reshape(-1, offset.shape[0]) # [N ,8]
    return unique_coords, center2corner

