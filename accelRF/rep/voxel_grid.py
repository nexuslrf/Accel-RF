from typing import List, Optional, Tuple
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from .utils import discretize_points, offset_points, trilinear_interp
from .base import Explicit3D
from .._C.rep import _ext

MAX_DEPTH = 10000.0

class VoxelGrid(Explicit3D):
    '''
    Let's start with a simple voxel grid.
    '''
    def __init__(
        self,
        bbox: Tensor,
        voxel_size: float,
        use_corner: bool=True,
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
        self.use_corner = use_corner
        self.bbox = bbox
        v_min, v_max = bbox[:3], bbox[3:]
        steps = ((v_max - v_min) / voxel_size).round().long() + 1
        # note the difference between torch.meshgrid and np.meshgrid.
        center_coords = torch.stack(torch.meshgrid([torch.arange(s) for s in steps]), -1) # s_x,s_y,s_z,3
        center_points = (center_coords * voxel_size + v_min).reshape(-1, 3) # start from lower bound
        # self.register_buffer('center_coords', center_coords)
        n_voxels = center_points.shape[0]
        occupancy = torch.ones(n_voxels, dtype=torch.bool) # occupancy's length unchanges unless splitting

        # corner points
        if use_corner:
            corner_shape = steps+1
            n_corners = corner_shape.prod().item()
            offset = offset_points().long() # [8, 3]
            corner1d = torch.arange(n_corners).reshape(corner_shape.tolist()) 
            center2corner = (center_coords[...,None,:] + offset).reshape(-1, 8, 3) # [..., 8,3]
            center2corner = corner1d[center2corner[...,0], center2corner[...,1], center2corner[...,2]] # [..., 8]
            self.register_buffer('center2corner', center2corner)
            self.register_buffer('n_corners', torch.tensor(n_corners))
        
        # keep min max voxels, for ray_intersection
        max_ray_hit = min(steps.sum().item(), n_voxels)
        # register_buffer for saving and loading.
        self.register_buffer('occupancy', occupancy)
        self.register_buffer('grid_shape', steps) # self.grid_shape = steps
        self.register_buffer('center_points', center_points)
        self.register_buffer('n_voxels', torch.tensor(n_voxels))
        self.register_buffer('max_ray_hit', torch.tensor(max_ray_hit))
        self.register_buffer('voxel_size', torch.tensor(voxel_size))

    def ray_intersect(self, rays_o: Tensor, rays_d: Tensor):
        '''
        Args:
            rays_o, Tensor, (N_rays, 3)
            rays_d, Tensor, (N_rays, 3)
        Return:
            pts_idx, Tensor, (N_rays, max_hit)
            t_near, t_far    (N_rays, max_hit)
        '''
        pts_idx_1d, t_near, t_far = _ext.aabb_intersect(
            rays_o.contiguous(), rays_d.contiguous(), 
            self.center_points.contiguous(), self.voxel_size, self.max_ray_hit)
        t_near.masked_fill_(pts_idx_1d.eq(-1), MAX_DEPTH)
        t_near, sort_idx = t_near.sort(dim=-1)
        t_far = t_far.gather(-1, sort_idx)
        pts_idx_1d = pts_idx_1d.gather(-1, sort_idx)
        hits = pts_idx_1d.ne(-1).any(-1)
        return pts_idx_1d, t_near, t_far, hits

    # def get_corner_points(self, center_idx):
    #     corner_idx = self.center2corner[center_idx] # [..., 8]
    #     return self.corner_points[corner_idx] # [..., 8, 3]

    def pruning(self, keep):
        n_vox_left = keep.sum()
        if n_vox_left > 0 and n_vox_left < keep.shape[0]:
            self.center_points = self.center_points[keep].contiguous()
            self.occupancy.masked_scatter_(self.occupancy, keep)
            self.n_voxels = n_vox_left
            self.max_ray_hit = self.get_max_ray_hit()
            if self.use_corner:
                c2corner_idx = self.center2corner[keep] # [..., 8]
                corner_idx, center2corner = c2corner_idx.unique(sorted=True, return_inverse=True) # [.] and [..., 8]
                self.center2corner = center2corner.contiguous()
                self.n_corners = self.n_corners * 0 + corner_idx.shape[0]
                return corner_idx

    def splitting(self, feats: Optional[Tensor]=None):
        offset = offset_points(device=self.center_points.device).long() # [8, 3] scale [0,1]
        n_subvox = offset.shape[0] # 8
        old_center_coords = discretize_points(self.center_points, self.voxel_size) # [N ,3]

        self.voxel_size *= 0.5
        half_voxel = self.voxel_size * 0.5
        self.center_points = (self.center_points[:,None,:] + (offset*2-1) * half_voxel).reshape(-1, 3)
        self.n_voxels = self.n_voxels * n_subvox
        self.grid_shape = self.grid_shape * 2
        self.occupancy = self.occupancy[...,None].repeat_interleave(n_subvox, -1).reshape(-1)
        self.max_ray_hit = self.get_max_ray_hit()
        if self.use_corner:
            center_coords = (2*old_center_coords[...,None,:] + offset).reshape(-1, 3) # [8N, 3] # x2
            # <==> discretize_points(self.center_points, self.voxel_size) # [8N ,3]
            corner_coords = (center_coords[...,None,:] + offset).reshape(-1, 3) # [64N, 3]
            unique_corners, center2corner = torch.unique(corner_coords, dim=0, sorted=True, return_inverse=True)
            self.n_corners = self.n_corners * 0 + unique_corners.shape[0]
            old_ct2cn = self.center2corner
            self.center2corner = center2corner.reshape(-1, n_subvox)
            if feats is not None:
                cn2oldct = center2corner.new_zeros(self.n_corners).scatter_(
                    0, center2corner, torch.arange(corner_coords.shape[0], device=feats.device) // n_subvox**2)
                feats_idx = old_ct2cn[cn2oldct] # [N_cn, 8]
                _feats = feats[feats_idx] # [N_cn, 8, D_f]
                new_feats = trilinear_interp(unique_corners-1, 2*old_center_coords[cn2oldct], _feats, 2., offset)
                return new_feats

    def get_max_ray_hit(self):
        # keep min max voxels, for ray_intersection
        min_voxel = self.center_points.min(0)[0]
        max_voxel = self.center_points.max(0)[0]
        aabb_box = ((max_voxel - min_voxel) / self.voxel_size).round().long() + 1
        max_ray_hit = min(aabb_box.sum(), self.n_voxels)
        return max_ray_hit
    
    def load_adjustment(self, n_voxels, grid_shape):
        self.center_points = self.center_points.new_empty(n_voxels, 3)
        self.center2corner = self.center2corner.new_empty(n_voxels, 8)
        self.occupancy = self.occupancy.new_empty(torch.tensor(grid_shape).prod())

    def get_edge(self):
        NotImplemented
        # TODO