from typing import List
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from .._C.rep import _ext

MAX_DEPTH = 10000.0

class VoxelGrid(nn.Module):
    '''
    Let's start with a simple dense voxel grid.
    '''
    def __init__(
        self,
        bbox: Tensor,
        voxel_size: float,
        use_corner: bool=True,
        device: torch.device='cuda'
        ):
        '''
        bbox2voxel: https://github.com/facebookresearch/NSVF/fairnr/modules/encoder.py#L1053
        bbox: array [min_x,y,z, max_x,y,z]

        x represents center, O represents corner
            O O O O O
             x x x x 
            O O O O O
             x x x x 
            O O O O O
        Given a center x's coords [i,j,k]. its corners' coords are [i,j,k] + {0,1}^3
        '''
        super().__init__()
        self.voxel_size = voxel_size
        self.device = device
        v_min, v_max = bbox[:3], bbox[3:]
        steps = ((v_max - v_min) / voxel_size).round().long() + 1
        # note the difference between torch.meshgrid and np.meshgrid.
        center_coords = torch.stack(torch.meshgrid([torch.arange(s) for s in steps]), -1) # s_x,s_y,s_z,3
        center_points = center_coords * voxel_size + v_min # start from lower bound
        # self.center_coords = center_coords.to(device)
        # self.center_points = center_points.to(device)
        self.register_buffer('center_coords', center_coords)
        self.register_buffer('center_points', center_points)
        # corner points
        if use_corner:
            corner_coords = torch.stack(torch.meshgrid([torch.arange(s+1) for s in steps]), -1)
            corner_points = corner_coords * voxel_size + v_min - 0.5 * voxel_size
            # self.corner_coords = corner_coords.to(device)
            # self.corner_points = corner_points.to(device)
            self.register_buffer('corner_coords', corner_coords)
            self.register_buffer('corner_points', corner_points)

    def ray_intersect(self, rays_o: Tensor, rays_d: Tensor):
        '''
        Args:
            rays_o, Tensor, (N_rays, 3)
            rays_d, Tensor, (N_rays, 3)
        Return:
            pts_idx?
            min_depth?
            max_depth?
        '''
        max_hit = sum(self.center_points.shape[:3])
        pts_idx_1d, min_depth, max_depth = _ext.aabb_intersect(
            rays_o.contiguous(), rays_d.contiguous(), 
            self.center_points.contiguous(), self.voxel_size, max_hit)
        min_depth.masked_fill_(pts_idx_1d.eq(-1), MAX_DEPTH)
        min_depth, sort_idx = min_depth.sort(dim=-1)
        max_depth = max_depth.gather(-1, sort_idx)
        pts_idx_1d = pts_idx_1d.gather(-1, sort_idx)
        hits = pts_idx_1d.ne(-1).any(-1)
        return pts_idx_1d, min_depth, max_depth, hits




        

        

