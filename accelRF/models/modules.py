from collections import OrderedDict
import torch
from torch import Tensor
import torch.nn as nn
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, include_input=True, N_freqs=4,
                log_sampling=True, freq_last=False):
        super().__init__()
        self.include_input = include_input # input at first 
        self.freq_last = freq_last

        if log_sampling:
            self.freq_bands = 2.**torch.linspace(0., (N_freqs-1), steps=N_freqs)[None,:,None]
        else:
            self.freq_bands = torch.linspace(2.**0., 2.**(N_freqs-1), steps=N_freqs)[None,:,None]
        
        self.period_offset = np.pi / 2 * torch.arange(2).reshape(1,1,2,1)
        self.freq_bands = nn.parameter.Parameter(self.freq_bands, requires_grad=False)
        self.period_offset = nn.parameter.Parameter(self.period_offset, requires_grad=False)

    # @torch.no_grad()
    def forward(self, x):
        bs, input_dims = x.shape
        
        embed = torch.sin(
            (x[:,None,:] * self.freq_bands).unsqueeze(-2) + self.period_offset
            ).reshape(bs, -1)

        # <==>
        # torch.cat([
        #     torch.sin(x[:,None,:] * self.freq_bands), 
        #     torch.cos(x[:,None,:] * self.freq_bands)
        # ], -1).reshape(bs, -1)

        if self.include_input:
            embed = torch.cat([x,embed], 1)
        if self.freq_last:
            embed = embed.reshape(bs, -1, input_dims).transpose(-1,-2).reshape(bs, -1)

        return embed
    
    # add two fake state_dict function to bypass saving/loading PE's parameters
    def state_dict(self, *args, **kwargs):
        return OrderedDict()

    def load_state_dict(self, *args, **kwargs):
        pass

    
