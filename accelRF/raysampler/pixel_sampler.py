from typing import Optional
import torch
import numpy as np
import torch.utils.data as data
from ..datasets import BaseDataset
from .utils import get_rays

class BasePixSampler(data.Dataset):
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
        output['uv'] = self.uv  # (N, 3)
        idx = torch.tensor(index)  # (N)
        output['idx'] = idx.expand(self.uv.shape[0])
        if 'gt_img' in img_dict:
            output['gt_rgb'] = img_dict['gt_img'].reshape(-1,3) # (N, 3)

        return output

class RenderingPixSampler(BasePixSampler):
    '''
    Just a alias of BasePixSampler.
    '''
    def __init__(
        self, dataset: BaseDataset, N_rand: int=0, device: torch.device = 'cpu', 
        rank: int = -1, n_replica: int = 1, seed: Optional[int] = None) -> None:

        super().__init__(dataset, N_rand, length=None, device=device, 
            rank=rank, n_replica=n_replica, seed=seed)

class BatchingPixSampler(BasePixSampler):
    def __init__(
        self, 
        dataset, 
        N_rand: int=2048,
        length: int=1000,
        device: torch.device='cpu',
        start_epoch: int=0,
        rank: int=-1,
        n_replica: int=1,
        seed: Optional[int]=None
    ) -> None:
        
        super().__init__(
            dataset, N_rand, length-start_epoch, False, device, rank, n_replica, seed)
        #
        H, W, _ = self.dataset.get_hwf()
        
        self.n_imgs = len(dataset)
        self.img_inds = torch.arange(self.n_imgs, device=device)[:,None].expand(-1, H*W).reshape(-1)

        self.n_pixels = self.n_imgs * H * W
        # For random ray batching
        self.shuffle_inds = torch.randperm(self.n_pixels, generator=self.rng, device=device)

        self.i_batch = 0

    def __getitem__(self, index):
        '''
        Return:
            rays_o: Tensor, sampled ray origins, [N_rays, 3]
            rays_d: Tensor, sampled ray directions, [N_rays, 3]
            gt_rgb: Tensor, ground truth color superized learning [N_rays, 3]
        '''
        # Random over all images
        batch_inds = self.shuffle_inds[self.i_batch+self.rank*self.N_rand:self.i_batch+(self.rank+1)*self.N_rand]
        uv = self.uv[batch_inds] # [B, 2]
        img_idx = self.img_inds[batch_inds]
        _uv = uv.long()
        rgb = self.dataset.imgs[img_idx, _uv[:,1], _uv[:,0]]

        output = {'uv': uv, 'idx': img_idx, 'gt_rgb': rgb}

        self.i_batch += self.N_rand * self.n_replica
        if self.i_batch >= self.rays_rgb.shape[1]:
            print("Shuffle data after an epoch!")
            self.shuffle_inds = torch.randperm(self.rays_rgb.shape[1], generator=self.rng, device=self.device)
            self.i_batch = 0

        return output

class PerViewPixSampler(BasePixSampler):
    
    def __init__(
        self, 
        dataset, 
        N_rand: int=2048,
        length: int=1000,
        N_views: int=1,
        precrop: bool=False,
        precrop_frac: float=0.5,
        precrop_iters: int=500,
        full_rays: bool=False,
        device: torch.device='cpu',
        start_epoch: int=0,
        rank: int=-1,
        n_replica: int=1,
        seed: Optional[int]=None
    ) -> None:
        
        super().__init__(
            dataset, N_rand, length-start_epoch, device, rank, n_replica, seed)
        self.N_views = N_views
        self.precrop = precrop
        self.precrop_frac = precrop_frac
        self.precrop_iters = precrop_iters - start_epoch
        self.full_rays = full_rays
        #
        H, W, _ = self.dataset.get_hwf()
        if full_rays:
            self.N_rand = (H*W - 1) // self.n_replica + 1
        # the current solution for iters after `precrop_iters`
        if self.precrop and self.precrop_iters > 0: 
            dH = int(H//2 * self.precrop_frac)
            dW = int(W//2 * self.precrop_frac)
            self.uv = torch.stack(
                torch.meshgrid(
                    torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH, device=self.device), 
                    torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW, device=self.device)
                ), -1).flip(-1).reshape(-1, 2)

    def disable_precrop(self) -> None: 
        '''
        user *can* disable precrop after K iters when Samper is warpped by DataLoader
        simply call `raysampler.disable_precrop` (instead of `dataloader.dataset.disable_precrop`)
        it will take effect after the current iteration (of length `self.length` due to prefetching)
        '''
        self.precrop = False
        H, W, _ = self.dataset.get_hwf()
        self.uv = torch.stack(
            torch.meshgrid(
                torch.linspace(0, H-1, H, device=self.device), 
                torch.linspace(0, W-1, W, device=self.device)
            ), -1).flip(-1).reshape(-1, 2) # (H, W, 2)

    def __getitem__(self, index):
        '''
        Return:
            rays_o: Tensor, sampled ray origins, [N_rays, 3]
            rays_d: Tensor, sampled ray directions, [N_rays, 3]
            gt_rgb: Tensor, ground truth color superized learning [N_rays, 3]
        '''
        output = {'uv': [], 'idx': [], 'gt_rgb': []}
        if self.precrop and index >= self.precrop_iters:
            self.disable_precrop()
            print('disable precrop!')
        for _ in range(self.N_views):
            img_i = torch.randint(len(self.dataset), (), generator=self.rng, device=self.device)
            target = self.dataset.imgs[img_i] # if 'gt_img' in img_dict else None
            idx = torch.tensor(img_i)
            if not self.full_rays:
                # To avoid manually setting numpy random seed for ender user when num_workers > 1, 
                # replace np.random.choice with torch.randperm
                # np.random.choice(self.coords.shape[0], size=[self.N_rand], replace=False)
                rand_inds = torch.randperm(self.uv.shape[0], generator=self.rng, device=self.device) # (len_shape, 1)
                select_inds = rand_inds[self.rank*self.N_rand:(self.rank+1)*self.N_rand]  # (N_rand,)
                uv = self.uv[select_inds]  # (N_rand, 2)
                select_coords = uv.flip(-1).long()
                
                output['uv'].append(uv)  # (N_rand, 2)
                output['idx'].append(idx.expand(uv.shape[0])) # (N_rand)
                output['gt_rgb'].append(target[select_coords[:, 0], select_coords[:, 1]])  # (N_rand, 3)
            else:
                output['uv'].append(self.uv[self.rank*self.N_rand:(self.rank+1)*self.N_rand])  # (N_rand, 3)
                output['idx'].append(idx.expand(output['uv'].shape[0])) # (N_rand)
                output['gt_rgb'].append(target.reshape(-1, 3)[self.rank*self.N_rand:(self.rank+1)*self.N_rand])  # (N_rand, 3)
        output = {k: torch.cat(output[k], 0) for k in output} # (N_views*N_rand, 3)
        # output['coords'] = self.coords # just for debug
        return output