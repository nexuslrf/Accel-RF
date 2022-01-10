from typing import List
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn

class VoxelGrid(nn.Module):
    '''
    Let's start with a simple dense voxel grid.
    '''
    def __init__(
        self,
        bbox: Tensor,
        voxel_size: float,
        use_corner: bool=True
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
        v_min, v_max = bbox[:3], bbox[3:]
        steps = ((v_max - v_min) / voxel_size).round().long() + 1
        # note the difference between torch.meshgrid and np.meshgrid.
        self.center_coords = torch.stack(torch.meshgrid([torch.arange(s) for s in steps]), -1) # s_x,s_y,s_z,3
        self.center_points = self.center_coords * voxel_size + v_min # start from lower bound
        # corner points
        if use_corner:
            self.corner_coords = torch.stack(torch.meshgrid([torch.arange(s+1) for s in steps]), -1)
            self.corner_points = self.corner_coords * voxel_size + v_min - 0.5 * voxel_size

    def ray_intersect(self, rays_o, rays_d):
        '''
        Args:
            rays_o,
            rays_d,
        Return:
            pts_idx?
            min_depth?
            max_depth?
        '''
        NotImplemented

        




        

        

