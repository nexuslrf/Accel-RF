import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


class PositionalEncoding(nn.Module):
    def __init__(self, include_input=True, N_freqs=4, log_sampling=True, 
        angular_enc=False, freq_last=False):
        super().__init__()
        self.include_input = include_input # input at first 
        self.angular_enc = angular_enc
        self.freq_last = freq_last

        if log_sampling:
            freq_bands = 2.**torch.linspace(0., (N_freqs-1), steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**(N_freqs-1), steps=N_freqs)
        
        self.half_pi = np.pi / 2
        self.register_buffer('freq_bands', freq_bands, False) # no need to checkpoint

    @torch.no_grad()
    def forward(self, x: Tensor):
        if self.angular_enc: 
            x = torch.acos(x.clamp(-1 + 1e-6, 1 - 1e-6)) # used in NSVF..
        fx = torch.einsum('...c,f->...fc', x, self.freq_bands) # einsum 🥰 🥰 🥰 
        embed = torch.sin(torch.cat([fx, fx + self.half_pi], -2)) # [..., 2*N_freqs, in_ch]
        # <==>
        # torch.cat([torch.sin(fx), torch.cos(fx)], -1)

        if self.include_input:
            embed = torch.cat([x.unsqueeze(-2),embed], -2) # [..., 2*N_freqs+1, in_ch]
        if self.freq_last:
            embed = embed.transpose(-1,-2) # [..., in_ch, 2*N_freqs?(+1)]

        embed = embed.flatten(-2) # [..., in_ch * ( 2*N_freqs?(+1) )]
        return embed