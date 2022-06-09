from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor
from ..datasets import BaseDataset
from .utils import get_rays_uv

'''
Wrapping get_rays into a independent module. Reasons:
1. to support trainable camera intrinsics & extrinsics, and make it more formal.
2. offload computation-heavy get_rays from CPU to GPU. (a bottleneck causing low util. of GPU)
'''

class CameraRays(nn.Module):
    '''
    '''
    def __init__(
        self,
        normalize_dir: bool=False, openGL_coord: bool=True
    ):
        super().__init__()
        self.normalize_dir = normalize_dir
        self.openGL_coord = openGL_coord


    def forward(self, 
        uv: Tensor, 
        idx: Optional[Tensor], dataset: Optional[BaseDataset],
        gt_rgb: Optional[Tensor],
        K: Optional[Tensor]=None, T: Optional[Tensor]=None, 
    ):
        '''
        This is a very flexible function.
        Inputs:
            uv: sampled coords on img plane, [N, 2]
            idx: uv sample's img idx, used to retrieve K&E stored in this class.
            K: camera intrinsics [N|1, x_shape]
                x_shape: 
                    - [3] --> H, W, focal
                    - [4] --> H, W, focalx, focaly
                    - [3,3] --> K matrix
            T: camera extrinsics [N|1, 4, 4] or [N|1, 4, 3]

        if K & T are not None, idx will be ignored.
        '''
        if K is None and T is None:
            K = dataset.Ks[idx] if dataset.Ks.shape[0] > 1 else dataset.Ks
            T = dataset.Ts[idx]

        if uv.device != K.device:
            uv = uv.to(K.device)
            gt_rgb = gt_rgb.to(K.device) if gt_rgb is not None else None

        rays_o, rays_d = get_rays_uv(uv, K, T, self.normalize_dir, self.openGL_coord)

        return rays_o, rays_d, gt_rgb