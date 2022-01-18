import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from ..rep import Explicit3D
from ..rep.utils import trilinear_interp

class PositionalEncoding(nn.Module):
    def __init__(self, N_freqs=4, include_input=True, log_sampling=True, 
        angular_enc=False, pi_bands=False, freq_last=False):
        super().__init__()
        self.include_input = include_input # input at first 
        self.angular_enc = angular_enc
        self.pi_bands = pi_bands
        self.freq_last = freq_last

        if log_sampling:
            freq_bands = 2.**torch.linspace(0., (N_freqs-1), steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**(N_freqs-1), steps=N_freqs)

        if pi_bands:
            freq_bands = freq_bands * np.pi
        self.half_pi = np.pi / 2
        self.register_buffer('freq_bands', freq_bands, False) # no need to checkpoint

    # @torch.no_grad()
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
    def __init__(self, n_embeds: int, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.embeddings = nn.Embedding(n_embeds, embed_dim)
        nn.init.normal_(self.embeddings.weight, mean=0, std=embed_dim ** -0.5)
        interp_offset = torch.stack(torch.meshgrid([torch.tensor([0.,1.])]*3),-1).reshape(-1,3)
        self.register_buffer('interp_offset', interp_offset)

    def forward(self, pts: Tensor, p2v_idx: Tensor, vox_rep: Explicit3D, per_voxel: bool=False):
        '''
        if per_voxel is False
            Args:
                pts: Tensor, [N_pts, 3]
                p2v_idx: Tensor, [N_pts] 
                    mapping pts to voxel idx, Note voxel idx are 1D index, and -1 idx should be masked out.
        if per_voxel is True
            Args:
                pts: Tensor, relative points in one voxel, [N_pts_per_vox, 3], scale [0,1]
                p2v_idx: Tensor, [N_vox]
            Return:
                embeds, [N_vox, N_pts_per_vox, embed_dim]
        '''
        # get corresponding voxel embeddings
        p2v_idx = p2v_idx.long()
        center_pts = vox_rep.center_points[p2v_idx] # (N, 3)
        corner_idx = vox_rep.center2corner[p2v_idx] # (N, 8)
        embeds = self.embeddings(corner_idx) # (N, 8, embed_dim)
        
        # interpolation
        if not per_voxel:
            interp_embeds = trilinear_interp(pts, center_pts, embeds, 
                vox_rep.voxel_size, self.interp_offset)
        else:
            pts = pts[...,None,:] # [N_pts_per_vox, 1, 3]
            r = (pts*self.interp_offset + (1-pts)*(1-self.interp_offset))\
                    .prod(dim=-1, keepdim=True)[None,:] # [1, N_ppv, 8, 1]
            interp_embeds = (embeds[:,None,:] * r).sum(-2) # [N_v, N_ppv, embed_dim]
        return interp_embeds

    def update_embeddings(self, new_embeddings):
        # https://stackoverflow.com/a/55766749/14835451
        n_emb = new_embeddings.shape[0]
        self.embeddings = nn.Embedding.from_pretrained(new_embeddings, freeze=False)
    
    def load_adjustment(self, num_embeds):
        w = self.embeddings.weight
        new_w = w.new_empty(num_embeds, self.embed_dim)
        self.embeddings = nn.Embedding.from_pretrained(new_w, freeze=False)
    
    def get_weight(self):
        return self.embeddings.weight