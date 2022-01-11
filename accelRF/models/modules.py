import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


class PositionalEncoding(nn.Module):
    def __init__(self, include_input=True, N_freqs=4,
                log_sampling=True, freq_last=False):
        super().__init__()
        self.include_input = include_input # input at first 
        self.freq_last = freq_last

        if log_sampling:
            freq_bands = 2.**torch.linspace(0., (N_freqs-1), steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**(N_freqs-1), steps=N_freqs)
        
        self.half_pi = np.pi / 2
        self.register_buffer('freq_bands', freq_bands, False) # no need to checkpoint

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

def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    return m
    

class PosEmbLinear(nn.Module):

    def __init__(self, in_dim, out_dim, no_linear=False, scale=1, *args, **kwargs):
        super().__init__()
        assert out_dim % (2 * in_dim) == 0, "dimension must be dividable"
        half_dim = out_dim // 2 // in_dim
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        
        self.emb = nn.Parameter(emb, requires_grad=False)
        self.linear = Linear(out_dim, out_dim) if not no_linear else None
        self.scale = scale
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.cat_input = False

    def forward(self, x):
        assert x.size(-1) == self.in_dim, "size must match"
        sizes = x.size()
        x = self.scale * x.unsqueeze(-1) @ self.emb.unsqueeze(0)
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        x = x.view(*sizes[:-1], self.out_dim)
        if self.linear is not None:
            return self.linear(x)
        return x


class NeRFPosEmbLinear(nn.Module):

    def __init__(self, in_dim, out_dim, angular=False, no_linear=False, cat_input=False):
        super().__init__()
        assert out_dim % (2 * in_dim) == 0, "dimension must be dividable"
        L = out_dim // 2 // in_dim
        emb = torch.exp(torch.arange(L, dtype=torch.float) * math.log(2.)) # just 1, 2, 4, 8, 16, 32, no need to make it complicated...
        if not angular:
            emb = emb * math.pi

        self.emb = nn.Parameter(emb, requires_grad=False)
        self.angular = angular
        self.linear = Linear(out_dim, out_dim) if not no_linear else None
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.cat_input = cat_input

    def forward(self, x):
        assert x.size(-1) == self.in_dim, "size must match"
        sizes = x.size() 
        inputs = x.clone()

        if self.angular:
            x = torch.acos(x.clamp(-1 + 1e-6, 1 - 1e-6))
        x = x.unsqueeze(-1) @ self.emb.unsqueeze(0)
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        x = x.view(*sizes[:-1], self.out_dim)
        if self.linear is not None:
            x = self.linear(x)
        if self.cat_input:
            x = torch.cat([x, inputs], -1)
        return x

    def extra_repr(self) -> str:
        outstr = 'Sinusoidal (in={}, out={}, angular={})'.format(
            self.in_dim, self.out_dim, self.angular)
        if self.cat_input:
            outstr = 'Cat({}, {})'.format(outstr, self.in_dim)
        return outstr
    
