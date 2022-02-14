from typing import Dict, List, Tuple, Optional
import copy
import torch
import torch.utils.data as data
from torch import Tensor

class BaseDataset(data.Dataset):
    Ks: Tensor  # camera intrinsics
    Ts: Tensor  # camera extrinsics
    img_paths: List[str]
    pixel_masks: Optional[Tensor]
    
    def __init__(self) -> None:
        super().__init__()
        self.downsample_factor = 1
        self.openGL_coord = True
        self.pixel_masks = None
        self.with_bbox = False
        self.device = 'cpu'

    def render_downsample(self, render_factor: int):
        self.downsample_factor = render_factor

    def get_hwf(self) -> Tuple:
        f = self.downsample_factor
        return self.H//f, self.W//f, self.focal/f

    def get_K(self, idx) -> Tensor:
        K = self.Ks[idx] if self.Ks.shape[0] > 1 else self.Ks[0]
        return K / self.downsample_factor

    def get_pixel_masks(self) -> Tensor:
        if self.pixel_masks is None:
            self.pixel_masks = torch.ones(*self.imgs.shape[:3], dtype=torch.bool)
            if self.with_bbox:
                from ..raysampler.utils import aabb_intersect_bbox, get_rays_uv
                H, W, _ = self.get_hwf()
                uv = torch.stack(
                        torch.meshgrid(
                            torch.linspace(0, H-1, H, device=self.device), 
                            torch.linspace(0, W-1, W, device=self.device)
                        ), -1).flip(-1).reshape(-1, 2) # (N, 2)
                for i in range(len(self.Ts)):
                    K = self.get_K(i)[None,:]
                    rays_o, rays_d = get_rays_uv(uv, K, self.Ts[i:i+1], openGL_coord=self.openGL_coord)
                    tnear, tfar = aabb_intersect_bbox(rays_o, rays_d, self.bbox, self.near, self.far)
                    hits = tfar >= tnear
                    self.pixel_masks[i] = hits.reshape(*self.pixel_masks.shape[1:])
        return self.pixel_masks

    def __len__(self) -> int:
        return len(self.Ts)    

    # this function can be overwriten
    def __getitem__(self, index) -> Dict:
        if self.imgs is not None:
            return {
                'extrinsics': self.Ts[index],
                'gt_img': self.imgs[index]
            }
        else:
            return {
                'extrinsics': self.Ts[index]
            }
    
    def to(self, device, include_rgb: bool=False):
        self.device = device

        self.Ks = self.Ks.to(device)
        self.Ts = self.Ts.to(device)

        if include_rgb and self.imgs is not None:
            self.imgs = self.imgs.to(device)
        return self

    def get_sub_set(self, split_set: str):
        sub_set = copy.copy(self)
        sub_set.imgs = sub_set.imgs[self.i_split[split_set]]
        sub_set.Ts = sub_set.Ts[self.i_split[split_set]]
        sub_set.img_paths = sub_set.img_paths[self.i_split[split_set]]
        if sub_set.Ks.shape[0] > 1:
            sub_set.Ks = sub_set.Ks[self.i_split[split_set]]
        return sub_set