from typing import Dict, List, Tuple
import copy
import torch
import torch.utils.data as data
from torch import Tensor

class BaseDataset(data.Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.downsample_factor = 1
        self.GL_coord = True

    def render_downsample(self, render_factor: int):
        self.downsample_factor = render_factor

    def get_hwf(self) -> Tuple:
        f = self.downsample_factor
        return self.H//f, self.W//f, self.focal/f

    def get_K(self) -> Tensor:
        '''
        Convert hwf into a 3x3 matrix
        '''
        return torch.tensor([
            [self.focal, 0, 0.5*self.W],
            [0, self.focal, 0.5*self.H],
            [0,          0,          1]
        ], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.poses)    

    # this function can be overwriten
    def __getitem__(self, index) -> Dict:
        if self.imgs is not None:
            return {
                'pose': self.poses[index],
                'gt_img': self.imgs[index]
            }
        else:
            return {'pose': self.poses[index]}
    
    def to(self, device):
        self.poses = self.poses.to(device)
        if self.imgs is not None:
            self.imgs = self.imgs.to(device)
        return self

    def get_sub_set(self, split_set: str):
        sub_set = copy.copy(self)
        sub_set.imgs = sub_set.imgs[self.i_split[split_set]]
        sub_set.poses = sub_set.poses[self.i_split[split_set]]
        return sub_set