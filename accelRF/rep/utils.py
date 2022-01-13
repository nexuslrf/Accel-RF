from typing import Optional
from numpy import interp
import torch
from torch import Tensor

@torch.jit.script
def bbox2voxels(bbox: Tensor, voxel_size: float):
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