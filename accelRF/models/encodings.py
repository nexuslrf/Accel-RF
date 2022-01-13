import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from ..rep import Explicit3D
from ..rep.utils import trilinear_interp

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

class VoxelEncoding(nn.Module):
    def __init__(self, vox_rep: Explicit3D, embed_dim: int):
        super().__init__()
        self.vox_rep = vox_rep
        self.embed_dim = embed_dim
        num_feat_pts = self.vox_rep.n_corners
        self.voxel_embeddings = nn.Embedding(num_feat_pts, embed_dim)
        nn.init.normal_(self.voxel_embeddings.weight, mean=0, std=embed_dim ** -0.5)
        interp_offset = torch.stack(torch.meshgrid([torch.tensor([0.,1.])]*3),-1).reshape(-1,3)
        self.register_buffer('interp_offset', interp_offset)

    def forward(self, pts: Tensor, p2v_idx: Tensor):
        '''
        Args:
            p2v_idx: Tensor, [N_pts] 
                mapping pts to voxel idx, Note voxel idx are 1D index, and -1 idx should be masked out.

        '''
        # get corresponding voxel embeddings
        vox_idx = p2v_idx.long()
        center_pts = self.vox_rep.center_points[vox_idx] # (N, 3)
        corner_idx = self.vox_rep.center2corner[vox_idx] # (N, 8)
        corner_pts = self.vox_rep.corner_points[corner_idx] # (N, 8, 3)
        embeds = self.voxel_embeddings(corner_idx) # (N, 8, embed_dim)
        
        # interpolation
        interp_embeds = trilinear_interp(pts, center_pts, embeds, 
            self.vox_rep.voxel_size, self.interp_offset)
        
        return interp_embeds

    def pruning(self):
        NotImplemented
    
    def splitting(self):
        NotImplemented
