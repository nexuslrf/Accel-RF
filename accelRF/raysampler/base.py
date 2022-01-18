from typing import Optional
import torch
import numpy as np
import torch.utils.data as data
from ..datasets import BaseDataset
from .utils import get_rays

class BaseRaySampler(data.Dataset):
    def __init__(
        self,
        dataset: Optional[BaseDataset],
        N_rand: int,
        length: Optional[int]=None,
        device: torch.device='cpu',
        rank: int=-1,
        n_replica: int=1,
        seed: Optional[int]=None
        ) -> None:
        super().__init__()
        self.dataset = dataset
        self.N_rand = N_rand
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

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index):
        output = {}
        img_dict = self.dataset[index]
        pose = img_dict['pose'][:3,:4]
        # cam_viewdir = img_dict['pose'][:3,2]
        rays_o, rays_d = get_rays(*self.dataset.get_hwf(), pose)
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
        self, dataset: BaseDataset, N_rand: int=0, device: torch.device = 'cpu', 
        rank: int = -1, n_replica: int = 1, seed: Optional[int] = None) -> None:

        super().__init__(dataset, N_rand, length=None, device=device, rank=rank, n_replica=n_replica, seed=seed)