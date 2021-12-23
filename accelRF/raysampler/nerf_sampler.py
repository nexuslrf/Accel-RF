import torch
import numpy as np
from .base import BaseSampler
from .utils import get_rays


class NeRFSampler(BaseSampler):
    """
    Ray sampling used in NeRF

    Args:
        -
        -
    """
    def __init__(
        self, 
        dataset, 
        N_rand: int=2048,
        length: int=32,
        use_batching: bool=False,
        precrop_frac: float=0) -> None:
        
        super().__init__(dataset, N_rand, length)
        self.use_batching = use_batching
        self.precrop_frac = precrop_frac
        self.precrop = precrop_frac > 0
        
        H, W, focal = self.dataset.get_hwf()

        if use_batching:
            # For random ray batching
            print('get rays')
            rays = torch.stack([get_rays(H, W, focal, p) for p in self.dataset.poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
            print('done, concats')
            rays_rgb = torch.cat([rays, self.dataset.images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
            rays_rgb = torch.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
            rays_rgb = torch.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
            # rays_rgb = rays_rgb.astype(np.float32)
            print('shuffle rays')
            rays_rgb = rays_rgb[torch.randperm(rays_rgb.shape[0])]
            print('done')
            i_batch = 0
        
        else:
            if self.precrop:
                dH = int(H//2 * self.precrop_frac)
                dW = int(W//2 * self.precrop_frac)
                self.coords = torch.stack(
                    torch.meshgrid(
                        torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                        torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                    ), -1)

        
    def reset(self) -> None:
        """
        reset random choices of length `self.queue_length`
        """
        self.choices = torch.randint(len(self.dataset), [self.queue_length])

    def disable_precrop(self) -> None:
        self.precrop = False
        H, W, _ = self.dataset.get_hwf()
        self.coords = torch.stack(
            torch.meshgrid(
                torch.linspace(0, H-1, H), 
                torch.linspace(0, W-1, W)
            ), -1)  # (H, W, 2)

    def __getitem__(self, index):
        pass