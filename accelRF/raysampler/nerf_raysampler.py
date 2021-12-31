import torch
import numpy as np
from .base import BaseRaySampler
from .utils import get_rays, ndc_rays


class NeRFRaySampler(BaseRaySampler):
    """
    Ray sampling used in NeRF, can work with Dataloader 

    Args:
        -
        -

    .. note:: 
        use_batching mode can only support num_workers <= 1
        dataloader shuffle must be False if (precrop==True and not use_batching)
    """
    # TODO add full image rendering mode
    def __init__(
        self, 
        dataset, 
        N_rand: int=2048,
        length: int=1000,
        use_batching: bool=False,
        use_ndc: bool=False,
        full_rendering: bool=False,
        precrop: bool=False,
        precrop_frac: float=0.5,
        precrop_iters: int=500,
        device: torch.device='cpu'
        ) -> None:
        
        super().__init__(dataset, N_rand, length, device)
        self.use_batching = use_batching
        self.use_ndc = use_ndc
        self.full_rendering = full_rendering
        self.precrop = precrop
        self.precrop_frac = precrop_frac
        self.precrop_iters = precrop_iters
        if full_rendering: 
            assert use_batching==False and precrop==False
            self.length = len(self.dataset)
        
        H, W, focal = self.dataset.get_hwf()

        if self.use_batching:
            # For random ray batching
            print('get rays')
            rays_o, rays_d = [], []
            for p in self.dataset.poses[:,:3,:4]:
                ray_o, ray_d = get_rays(H, W, focal, p)
                rays_o.append(ray_o); rays_d.append(ray_d)
            rays_o = torch.stack(rays_o, 0); rays_d = torch.stack(rays_d, 0) # [N, H, W, 3]
            print('done, concats')
            rays_rgb = torch.stack([rays_o, rays_d, self.dataset.imgs], 0) # [ro+rd+rgb, N, H, W, 3]
            self.rays_rgb = torch.reshape(rays_rgb, [3,-1,3]) # [ro+rd+rgb, N*H*W, 3]
            print('shuffle rays')
            self.shuffle_inds = torch.randperm(self.rays_rgb.shape[1], device=self.device)
            print('done')
            self.i_batch = 0
        
        else:
            # the current solution for iters after `precrop_iters`
            if self.precrop: 
                dH = int(H//2 * self.precrop_frac)
                dW = int(W//2 * self.precrop_frac)
                self.coords = torch.stack(
                    torch.meshgrid(
                        torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH, device=self.device), 
                        torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW, device=self.device)
                    ), -1)
            else:
                self.coords = torch.stack(torch.meshgrid(
                    torch.linspace(0, H-1, H, device=self.device), 
                    torch.linspace(0, W-1, W, device=self.device)), -1)  # (H, W, 2)
            self.coords = self.coords.reshape(-1, 2)

    def disable_precrop(self) -> None: 
        '''
        user *can* disable precrop after K iters when NeRFSamper is warpped by DataLoader
        simply call `raysampler.disable_precrop` (instead of `dataloader.dataset.disable_precrop`)
        it will take effect after the current iteration (of length `self.length` due to prefetching)
        '''
        self.precrop = False
        H, W, _ = self.dataset.get_hwf()
        if not self.use_batching:
            self.coords = torch.stack(
                torch.meshgrid(
                    torch.linspace(0, H-1, H, device=self.device), 
                    torch.linspace(0, W-1, W, device=self.device)
                ), -1).reshape(-1, 2) # (H, W, 2)

    def __getitem__(self, index):
        '''
        Return:
            rays_o: Tensor, sampled ray origins, [N_rays, 3]
            rays_d: Tensor, sampled ray directions, [N_rays, 3]
            gt_rgb: Tensor, ground truth color superized learning [N_rays, 3]
        '''
        output = {}
        if not self.full_rendering:
            if self.use_batching:
                # Random over all images
                batch_inds = self.shuffle_inds[self.i_batch:self.i_batch+self.N_rand]
                batch = self.rays_rgb[:, batch_inds] # [3, B, 2+1]
                output['rays_o'], output['rays_d'], output['gt_rgb'] = batch[0], batch[1], batch[2]
                cam_viewdir = None
                self.i_batch += self.N_rand
                if self.i_batch >= self.rays_rgb.shape[1]:
                    print("Shuffle data after an epoch!")
                    self.shuffle_inds = torch.randperm(self.rays_rgb.shape[1], device=self.device)
                    self.i_batch = 0
            else:
                if self.precrop and index >= self.precrop_iters:
                    self.disable_precrop()
                    print('disable precrop!')
                img_i = torch.randint(len(self.dataset), ())
                img_dict = self.dataset[img_i]
                pose = img_dict['pose'][:3,:4]
                cam_viewdir = img_dict['pose'][:3,2]
                target = img_dict['gt_img'] # if 'gt_img' in img_dict else None
                rays_o, rays_d = get_rays(*self.dataset.get_hwf(), pose)

                # To avoid manually setting numpy random seed for ender user when num_workers > 1, 
                # replace np.random.choice with torch.randperm
                # np.random.choice(self.coords.shape[0], size=[self.N_rand], replace=False)
                select_inds = torch.randperm(self.coords.shape[0])[:self.N_rand]  # (N_rand,)
                select_coords = self.coords[select_inds].long()  # (N_rand, 2)
                output['rays_o'] = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                output['rays_d'] = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                output['gt_rgb'] = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                output['coords'] = self.coords # just for debug
        else:
            img_dict = self.dataset[index]
            pose = img_dict['pose'][:3,:4]
            cam_viewdir = img_dict['pose'][:3,2]
            rays_o, rays_d = get_rays(*self.dataset.get_hwf(), pose)
            output['rays_o'] = rays_o.reshape(-1,3)  # (N, 3)
            output['rays_d'] = rays_d.reshape(-1,3)  # (N, 3)
            if 'gt_img' in img_dict:
                output['gt_rgb'] = img_dict['gt_img'].reshape(-1,3) # (N, 3)

        # # Update: remove use_viewdirs, this value can be computed later, thus reducing host-device mem comm.
        # if self.use_viewdirs:
        #     output['viewdirs'] = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        
        if self.use_ndc:
            # for forward facing scenes
            output['rays_o'], output['rays_d'] = \
                ndc_rays(self.dataset.get_hwf(), 1., rays_o, rays_d)

        return output





            

