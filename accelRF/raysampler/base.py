from typing import Optional
import torch
import numpy as np
import torch.utils.data as data
from ..datasets import BaseDataset
from .utils import get_rays_uv

class BaseRaySampler(data.Dataset):
    def __init__(
        self,
        dataset: Optional[BaseDataset],
        N_rand: int,
        length: Optional[int]=None,
        normalize_dir: bool=False,
        use_mask: bool=False,
        device: torch.device='cpu',
        rank: int=-1,
        n_replica: int=1,
        seed: Optional[int]=None
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.N_rand = N_rand
        self.normalize_dir = normalize_dir
        self.use_mask = use_mask
        self.length = length if length is not None else len(dataset)
        self.device = device

        # for distributed settings
        self.rng = torch.Generator(device=device)
        self.n_replica = n_replica
        self.rank = 0
        if rank >= 0:
            self.rank = rank
            self.rng.manual_seed(0)
        if seed is not None:
            self.rng.manual_seed(seed)
        
        H, W, _ = self.dataset.get_hwf()
        # Note: self.uv[x, y] = [y, x]
        self.uv = torch.stack(torch.meshgrid(
                        torch.linspace(0, H-1, H, device=self.device), 
                        torch.linspace(0, W-1, W, device=self.device)),
                    -1).flip(-1).reshape(-1, 2)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index):
        output = {}
        img_dict = self.dataset[index]
        T = img_dict['extrinsics'][None,...]
        K = self.dataset.get_K(index)[None,...]
        # cam_viewdir = img_dict['pose'][:3,2]
        rays_o, rays_d = get_rays_uv(self.uv, K, T, normalize_dir=self.normalize_dir, 
                                    openGL_coord=self.dataset.openGL_coord)
        output['rays_o'] = rays_o.reshape(-1,3)  # (N, 3)
        output['rays_d'] = rays_d.reshape(-1,3)  # (N, 3)
        if 'gt_img' in img_dict:
            output['gt_rgb'] = img_dict['gt_img'].reshape(-1,3) # (N, 3)

        return output


class RenderingRaySampler(BaseRaySampler):
    '''
    Just a alias of BaseRaySampler.
    '''
    def __init__(
        self, dataset: BaseDataset, N_rand: int=0, normalize_dir: bool=False, 
        device: torch.device = 'cpu', 
        rank: int = -1, n_replica: int = 1, seed: Optional[int] = None) -> None:

        super().__init__(dataset, N_rand, length=None, normalize_dir=normalize_dir,
                device=device, rank=rank, n_replica=n_replica, seed=seed)