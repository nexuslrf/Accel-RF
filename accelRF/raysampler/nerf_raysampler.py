from typing import Optional
import torch
import numpy as np
import torch.utils.data as data
from .base import BaseRaySampler, RenderingRaySampler
from .utils import get_rays_uv

class BatchingRaySampler(BaseRaySampler):
    def __init__(
        self, 
        dataset, 
        N_rand: int=2048,
        length: int=1000,
        normalize_dir: bool=False,
        precomp_rays: bool=False,
        use_mask: bool=False,
        device: torch.device='cpu',
        start_epoch: int=0,
        rank: int=-1,
        n_replica: int=1,
        seed: Optional[int]=None
        ) -> None:
        
        super().__init__(
            dataset, N_rand, length-start_epoch, normalize_dir, use_mask,
            device=device, rank=rank, n_replica=n_replica, seed=seed)
        #
        H, W, focal = self.dataset.get_hwf()
        self.precomp_rays = precomp_rays
        self.n_imgs = len(dataset)
        self.n_pixels = self.n_imgs * H * W
        self.n_per_batch = self.N_rand * self.n_replica
        # For random ray batching
        if precomp_rays:
            rays_o, rays_d = [], []
            for i in range(self.n_imgs):
                K = dataset.get_K(i)[None,:]
                ray_o, ray_d = get_rays_uv(self.uv, K, dataset.Ts[i:i+1], normalize_dir=normalize_dir, 
                                        openGL_coord=self.dataset.openGL_coord)
                rays_o.append(ray_o); rays_d.append(ray_d)
            rays_o = torch.stack(rays_o, 0); rays_d = torch.stack(rays_d, 0) # [N, H, W, 3]
            rays_rgb = torch.stack([rays_o, rays_d, self.dataset.imgs], 0) # [ro+rd+rgb, N, H, W, 3]
            self.rays_rgb = torch.reshape(rays_rgb, [3,-1,3]) # [ro+rd+rgb, N*H*W, 3]
        else:
            self.img_inds = torch.arange(self.n_imgs, device=device)[:,None].expand(-1, H*W).reshape(-1)

        self.reset_shuffle()
        self.i_batch = 0

    def reset_shuffle(self):
        self.shuffle_inds = torch.randperm(self.n_pixels, generator=self.rng, device=self.device)
        if self.use_mask:
            masks = self.dataset.get_pixel_masks().reshape(-1) # [n_pixels]
            self.shuffle_inds = self.shuffle_inds[masks]

    def __getitem__(self, index):
        '''
        Return:
            rays_o: Tensor, sampled ray origins, [N_rays, 3]
            rays_d: Tensor, sampled ray directions, [N_rays, 3]
            gt_rgb: Tensor, ground truth color superized learning [N_rays, 3]
        '''
        output = {}
        # Random over all images
        batch_inds = self.shuffle_inds[self.i_batch+self.rank*self.N_rand:self.i_batch+(self.rank+1)*self.N_rand]

        if self.precomp_rays:
            batch = self.rays_rgb[:, batch_inds] # [3, B, 2+1]
            output['rays_o'], output['rays_d'], output['gt_rgb'] = batch[0], batch[1], batch[2]
        else:
            uv = self.uv[batch_inds % self.uv.shape[0]] # [B, 2]
            img_idx = self.img_inds[batch_inds]
            _uv = uv.long()
            rgb = self.dataset.imgs[img_idx, _uv[:,1], _uv[:,0]]
            K = self.dataset.get_K(img_idx)[None,:]
            rays_o, rays_d = get_rays_uv(uv, K, self.dataset.Ts[img_idx], normalize_dir=self.normalize_dir, 
                                        openGL_coord=self.dataset.openGL_coord)
            output['rays_o'], output['rays_d'], output['gt_rgb'] = rays_o, rays_d, rgb
        if self.use_mask:
            output['pixel_idx'] = batch_inds

        self.i_batch += self.N_rand * self.n_replica
        if self.i_batch >= self.shuffle_inds.shape[0]:
            print("Shuffle data after an epoch!")
            self.reset_shuffle()
            self.i_batch = 0

        return output

class PerViewRaySampler(BaseRaySampler):
    
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
        normalize_dir: bool=False,
        use_mask: bool=False,
        device: torch.device='cpu',
        start_epoch: int=0,
        rank: int=-1,
        n_replica: int=1,
        seed: Optional[int]=None
        ) -> None:
        
        super().__init__(
            dataset, N_rand, length-start_epoch, normalize_dir, use_mask,
            device=device, rank=rank, n_replica=n_replica, seed=seed)
        self.N_views = N_views
        self.precrop = precrop
        self.precrop_frac = precrop_frac
        self.precrop_iters = precrop_iters - start_epoch
        self.full_rays = full_rays
        self.use_mask = use_mask
        #
        H, W, focal = self.dataset.get_hwf()
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
        user *can* disable precrop after K iters when NeRFSamper is warpped by DataLoader
        simply call `raysampler.disable_precrop` (instead of `dataloader.dataset.disable_precrop`)
        it will take effect after the current iteration (of length `self.length` due to prefetching)
        '''
        self.precrop = False
        H, W, _ = self.dataset.get_hwf()
        self.uv = torch.stack(
            torch.meshgrid(
                torch.linspace(0, H-1, H, device=self.device), 
                torch.linspace(0, W-1, W, device=self.device)
            ), -1).flip(-1).reshape(-1, 2) # (N, 2)


    def __getitem__(self, index):
        '''
        Return:
            rays_o: Tensor, sampled ray origins, [N_rays, 3]
            rays_d: Tensor, sampled ray directions, [N_rays, 3]
            gt_rgb: Tensor, ground truth color superized learning [N_rays, 3]
        '''
        output = {'rays_o': [], 'rays_d': [], 'gt_rgb': []}
        if self.precrop and index >= self.precrop_iters:
            self.disable_precrop()
            print('disable precrop!')
        
        if self.use_mask:
            mask = self.dataset.get_pixel_masks()
            pixels_per_img = mask.shape[1] * mask.shape[2]
            output['pixel_idx'] = []

        for _ in range(self.N_views):
            img_i = torch.randint(len(self.dataset), (), generator=self.rng, device=self.device)
            img_dict = self.dataset[img_i]
            T = img_dict['extrinsics'][None,:]
            K = self.dataset.get_K(img_i)[None,:]
            target = img_dict['gt_img'] # if 'gt_img' in img_dict else None
            
            view_uv = self.uv[mask[img_i].reshape(-1)] if self.use_mask else self.uv

            if not self.full_rays:
                # To avoid manually setting numpy random seed for ender user when num_workers > 1, 
                # replace np.random.choice with torch.randperm
                # np.random.choice(self.coords.shape[0], size=[self.N_rand], replace=False)
                rand_inds = torch.randperm(view_uv.shape[0], generator=self.rng, device=self.device) # (len_shape, 1)
                select_inds = rand_inds[self.rank*self.N_rand:(self.rank+1)*self.N_rand]  # (N_rand,)
                uv = view_uv[select_inds]  # (N_rand, 2)
            else:
                uv = view_uv[self.rank*self.N_rand:(self.rank+1)*self.N_rand]

            select_coords = uv.flip(-1).long()
            rays_o, rays_d = get_rays_uv(uv, K, T, normalize_dir=self.normalize_dir, 
                                openGL_coord=self.dataset.openGL_coord) # TODO optimize it
            output['rays_o'].append(rays_o)  # (N_rand, 3)
            output['rays_d'].append(rays_d)  # (N_rand, 3)
            output['gt_rgb'].append(target[select_coords[:, 0], select_coords[:, 1]])  # (N_rand, 3)
            if self.use_mask:
                output['pixel_idx'].append(pixels_per_img*img_i+select_coords[:,0]*mask.shape[2]+select_coords[:,1])

        output = {k: torch.cat(output[k], 0) for k in output} # (N_views*N_rand, 3)
        # output['coords'] = self.coords # just for debug
        return output
    

class NeRFRaySampler(data.Dataset):
    """
    Ray sampling used in NeRF, can work with Dataloader 
    This is an integration of BatchingRaySampler and PerViewRaySampler.
    Args:
        -
        -

    .. note:: 
        dataloader shuffle must be False if (precrop==True and not use_batching)
    """
    def __init__(
        self, 
        dataset, 
        N_rand: int=2048,
        length: int=1000,
        use_batching: bool=False,
        full_rendering: bool=False,
        precrop: bool=False,
        precrop_frac: float=0.5,
        precrop_iters: int=500,
        normalize_dir: bool=False,
        device: torch.device='cpu',
        start_epoch: int=0,
        rank: int=-1,
        n_replica: int=1,
        seed: Optional[int]=None
        ) -> None:
        
        super().__init__()
        self.dataset = dataset # just in case.
        
        if full_rendering: 
            assert use_batching==False and precrop==False
            self.length = len(self.dataset)
            self.raysampler = RenderingRaySampler(
                dataset, N_rand, False, device, rank, n_replica, seed
            )

        elif use_batching:
            # For random ray batching
            self.raysampler = BatchingRaySampler(
                dataset, N_rand, length, normalize_dir, False, False, 
                device=device, start_epoch=start_epoch, rank=rank, n_replica=n_replica, seed=seed
            )
        else:
            N_views = 1
            full_rays = False
            self.raysampler = PerViewRaySampler(
                dataset, N_rand, length, N_views, precrop, precrop_frac, precrop_iters, full_rays, 
                normalize_dir, 
                device=device, start_epoch=start_epoch, rank=rank, n_replica=n_replica, seed=seed
            )

    def __len__(self):
        return len(self.raysampler)

    def __getitem__(self, index):
        '''
        Return:
            rays_o: Tensor, sampled ray origins, [N_rays, 3]
            rays_d: Tensor, sampled ray directions, [N_rays, 3]
            gt_rgb: Tensor, ground truth color superized learning [N_rays, 3]
        '''
        output = self.raysampler[index]

        # # Update: remove use_viewdirs, this value can be computed later, thus reducing host-device mem comm.
        # if self.use_viewdirs:
        #     output['viewdirs'] = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        
        # # Update: after removing viewdirs from RaySampler, you cannot process ndc here.
        # if self.use_ndc: 
        #     # for forward facing scenes
        #     output['rays_o'], output['rays_d'] = \
        #         ndc_rays(*self.dataset.get_hwf(), 1., output['rays_o'], output['rays_d'])

        return output





            

