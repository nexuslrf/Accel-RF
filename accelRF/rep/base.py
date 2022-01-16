from typing import Tuple
import torch.nn as nn
from torch import Tensor

class Explicit3D(nn.Module):
    # corner_points: Tensor
    center_points: Tensor
    center2corner: Tensor
    n_voxels: int
    n_corners: int
    grid_shape: Tensor
    voxel_size: float
    occupancy: Tensor

    def __init__(self):
        super().__init__()
    
    def ray_intersect(
        self, rays_o: Tensor, rays_d: Tensor
        ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        NotImplemented
    
    def pruning(self, keep: Tensor):
        NotImplemented
    
    def splitting(self):
        NotImplemented