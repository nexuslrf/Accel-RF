from typing import Tuple
import torch.nn as nn
from torch import Tensor

class Explicit3D(nn.Module):
    def __init__(self):
        super().__init__()
    
    def ray_intersect(
        self, rays_o: Tensor, rays_d: Tensor
        ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        NotImplemented