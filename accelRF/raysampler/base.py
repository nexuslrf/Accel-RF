import torch
import numpy as np
import torch.utils.data as data
from ..datasets import BaseDataset

class BaseRaySampler(data.Dataset):
    def __init__(
        self,
        dataset: BaseDataset,
        N_rand: int,
        length: int,
        device: torch.device='cpu'
        ) -> None:
        super().__init__()
        self.dataset = dataset
        self.N_rand = N_rand
        self.length = length
        self.device = device

    def __len__(self) -> int:
        return self.length
