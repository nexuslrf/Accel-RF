from typing import List, Optional
import torch
import torch.nn as nn
from torch.tensor import Tensor

class NeRF(nn.Module):
    def __init__(self, 
        D: int=8, W: int=256,
        in_channels_pts: int=63, in_channels_dir: int=27,
        skips: List[int]=[4]
        ) -> None:
        """
        Args:
            D: number of layers for density (sigma) encoder
            W: number of hidden units in each layer
            in_channels_pts: number of input channels for pts (3+3*10*2=63 by default)
            in_channels_dir: number of input channels for direction (3+3*4*2=27 by default)
            skips: add skip connection in the Dth layer
        Note: not support use_viewdirs = False
        """
        super().__init__()
        self.D = D
        self.W = W
        self.in_channels_pts = in_channels_pts
        self.in_channels_dir = in_channels_dir
        self.skips = skips

        # pts encoding layers
        self.pts_layers = nn.ModuleList()
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels_pts, W)
            elif i-1 in skips:
                layer = nn.Linear(W+in_channels_pts, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            self.pts_layers.append(layer)
        # final layer
        self.pts_feature_layer = nn.Linear(W, W)

        # view direction encoding layers
        self.views_layers = nn.Sequential(
                                nn.Linear(W+in_channels_dir, W//2),
                                nn.ReLU(True))

        # output layers
        self.sigma_layer = nn.Linear(W, 1)
        self.rgb_layer = nn.Sequential(
                        nn.Linear(W//2, 3), # rgb_channel = 3
                        nn.Sigmoid()) # Note: sigmoid is already applied here!

    def forward(self, pts: Tensor, dir: Optional[Tensor]=None):
        """
        Encodes input (pts+dir) to rgb+sigma (not ready to render yet).
        For rendering this ray, please see rendering.py

        Args:
            pts: (B, self.in_channels_pts)
               the embedded vector of position and direction
            dir: (B, self.in_channels_dir)

        Outputs:
            Dict(
                sigma: (B, 1),
                rgb: (B, 3)
            )
        """
        pts_ = pts
        for i, layer in enumerate(self.pts_layers):
            pts_ = layer(pts_)
            if i in self.skips:
                pts_ = torch.cat([pts, pts_], -1)

        sigma = self.sigma_layer(pts_)

        if dir is None:
            return {'sigma': sigma}

        pts_feature = self.pts_feature_layer(pts_)

        dir_input = torch.cat([pts_feature, dir], -1)
        views_feature = self.views_layers(dir_input)
        rgb = self.rgb_layer(views_feature)

        return {
            'sigma': sigma,
            'rgb': rgb
        }
        # torch.cat([sigma, rgb], -1)