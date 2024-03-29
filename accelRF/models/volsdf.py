from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np

'''
Modules are basically follow the desing of the original repo, but also with some modifications.
https://github.com/lioryariv/volsdf/blob/main/code/model/network.py
'''

class SDFNet(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            sdf_bounding_sphere,
            d_in,
            d_out,
            dims,
            geometric_init=False,
            bias=0.0,
            skip_in=(),
            weight_norm=False,
            sphere_scale=1.0,
    ):
        super().__init__()

        self.sdf_bounding_sphere = sdf_bounding_sphere
        self.sphere_scale = sphere_scale
        dims = [d_in] + dims # + [d_out + feature_vector_size] 
        self.weight_norm = weight_norm
        self.num_layers = len(dims) - 2
        self.skip_in = skip_in

        self.lins = nn.ModuleList()
        for l in range(0, self.num_layers):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if d_in > 3 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif d_in > 3 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)
            
            self.lins.append(lin)

        # last layer.
        self.sdf_layer = nn.Linear(dims[-1], d_out)
        self.feat_layer = nn.Linear(dims[-1], feature_vector_size)

        for lin in [self.sdf_layer, self.feat_layer]:
            if geometric_init:
                torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                torch.nn.init.constant_(lin.bias, -bias)
            if weight_norm:
                lin = nn.utils.weight_norm(lin)

        self.softplus = nn.Softplus(beta=100)

    def forward(self, input: Tensor, xyz: Optional[Tensor]=None, sdf_only: bool=False):
        x = input
        for i, lin in enumerate(self.lins):
            if i in self.skip_in:
                x = torch.cat([x, input], -1) / (2**0.5)
            x = lin(x)
            x = self.softplus(x)
        sdf = self.sdf_layer(x)

        ''' Clamping the SDF with the scene bounding sphere, so that all rays are eventually occluded '''
        if self.sdf_bounding_sphere > 0.0 and xyz is not None:
            sphere_sdf = self.sphere_scale * (self.sdf_bounding_sphere - xyz.norm(2, -1, keepdim=True))
            sdf = torch.minimum(sdf, sphere_sdf)

        if sdf_only:
            return sdf, None
        else:
            return sdf, self.feat_layer(x)

class RGBNet(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            mode,
            d_in,
            d_out,
            dims,
            weight_norm=False,
    ):
        super().__init__()

        self.mode = mode
        dims = [d_in + feature_vector_size] + dims + [d_out]
        self.weight_norm = weight_norm
        self.num_layers = len(dims) - 1

        self.lins = nn.ModuleList()
        for l in range(0, self.num_layers):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)
            if weight_norm:
                lin = nn.utils.weight_norm(lin)
            self.lins.append(lin)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(
        self, view_dirs: Tensor, feature_vectors: Tensor, 
        points: Optional[Tensor]=None, normals: Optional[Tensor]=None
    ):
        if self.mode == 'idr':    # idr
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'nerf':                     # nerf
            rendering_input = torch.cat([view_dirs, feature_vectors], dim=-1)

        x = rendering_input

        for i, lin in enumerate(self.lins):
            x = lin(x)
            if i < self.num_layers - 1:
                x = self.relu(x)

        x = self.sigmoid(x)
        return x


######### Density #########

class Density(nn.Module):
    def __init__(self, beta=None):
        super().__init__()
        if beta is not None:
            self.beta = nn.Parameter(torch.tensor(beta))

    def forward(self, sdf, beta=None):
        return self.density_func(sdf, beta=beta)

def laplace_fn(sdf: Tensor, beta: Tensor):
    alpha = 1 / beta
    return alpha * (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() / beta))

class LaplaceDensity(Density):  # alpha * Laplace(loc=0, scale=beta).cdf(-sdf)
    def __init__(self, beta, beta_min=0.0001):
        super().__init__(beta)
        self.register_buffer('beta_min', torch.tensor(beta_min))

    def density_func(self, sdf, beta=None):
        if beta is None:
            beta = self.get_beta()

        alpha = 1 / beta
        return alpha * (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() / beta))

    def get_beta(self):
        beta = self.beta.abs() + self.beta_min
        return beta

class AbsDensity(Density):  # like NeRF++
    def density_func(self, sdf, beta=None):
        return torch.abs(sdf)


class SimpleDensity(Density):  # like NeRF
    def __init__(self, beta, noise_std=1.0):
        super().__init__(beta)
        self.noise_std = noise_std

    def density_func(self, sdf, beta=None):
        if self.training and self.noise_std > 0.0:
            noise = torch.randn(sdf.shape).cuda() * self.noise_std
            sdf = sdf + noise
        return torch.relu(sdf)


class NeuSDensity(nn.Module):
    def __init__(self, init_val):
        super().__init__()
        self.variance = nn.Parameter(torch.tensor(init_val))

    def forward(self, sdf, dirs, dists, gradients=None, cos_anneal_ratio=1.0):
        batch_size, n_samples = sdf.shape[0], sdf.shape[1]
        inv_s = torch.exp(self.variance * 10.0).clip(1e-6, 1e6)   # [1, 1]

        if gradients is not None:
            true_cos = (dirs * gradients).sum(-1, keepdim=True)

            # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
            # the cos value "not dead" at the beginning training iterations, for better convergence.
            iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                        F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

            # Estimate signed distances at section points
            estimated_next_sdf = sdf + iter_cos * dists[...,None] * 0.5
            estimated_prev_sdf = sdf - iter_cos * dists[...,None] * 0.5
        else:
            prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
            # prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
            mid_sdf = (prev_sdf + next_sdf) * 0.5
            cos_val = (next_sdf - prev_sdf) / (dists + 1e-5)
            prev_cos_val = torch.cat([torch.zeros([batch_size, 1]), cos_val[:, :-1]], dim=-1)
            cos_val = torch.min(cos_val, prev_cos_val)
            cos_val = cos_val.clip(-1e3, 0.0)

            estimated_prev_sdf = mid_sdf - cos_val * dists[...,None] * 0.5
            estimated_next_sdf = mid_sdf + cos_val * dists[...,None] * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).reshape(batch_size, n_samples).clip(0.0, 1.0)


        return alpha