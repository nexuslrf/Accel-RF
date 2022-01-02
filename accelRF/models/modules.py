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
            self.freq_bands = 2.**torch.linspace(0., (N_freqs-1), steps=N_freqs)
        else:
            self.freq_bands = torch.linspace(2.**0., 2.**(N_freqs-1), steps=N_freqs)
        
        self.half_pi = np.pi / 2
        self.freq_bands = nn.parameter.Parameter(self.freq_bands, requires_grad=False)

    @torch.no_grad()
    def forward(self, x: Tensor):
        fx = torch.einsum('...c,f->...fc', x, self.freq_bands) # einsum 🥰 🥰 🥰 
        embed = torch.sin(torch.cat([fx, fx + self.half_pi], -2)) # [..., 2*N_freqs, in_ch]
        # <==>
        # torch.cat([
        #     torch.sin(fx), 
        #     torch.cos(fx)
        # ], -1)

        if self.include_input:
            embed = torch.cat([x.unsqueeze(-2),embed], -2) # [..., 2*N_freqs+1, in_ch]
        if self.freq_last:
            embed = embed.transpose(-1,-2) # [..., in_ch, 2*N_freqs?(+1)]

        embed = embed.flatten(-2) # [..., in_ch * ( 2*N_freqs?(+1) )]
        return embed

    
