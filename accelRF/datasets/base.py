from typing import Dict, List, Tuple
import torch
import torch.utils.data as data

TRAIN = 0
VAL = 1
TEST = 2

class RFDataset(data.Dataset):
    def __init__(self) -> None:
        self.mode = TRAIN # choice in [TRAIN, VAL, TEST]

    def render_downsample(self, render_factor: int):
        self.downsample_factor = render_factor

    def get_hwf(self) -> Tuple:
        f = self.downsample_factor
        return self.H//f, self.W//f, self.focal/f


    def __getitem__(self, index) -> Dict:
        if self.mode == VAL:
            return {'pose': self.poses[self.mode][index]}
        elif self.mode == TEST:
            return {'pose': self.poses[self.mode][index], 'gt': self.imgs[self.mode][index]}

    # TODO ray batching function