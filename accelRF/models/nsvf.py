from typing import List, Optional, Tuple
import torch
import torch.nn as nn
from torch.tensor import Tensor

__all__ = ['NSVF_MLP', 'BackgroundField']

# start with two help nn.Modules

class FCLayer(nn.Module):
    """
    Reference:
        https://github.com/vsitzmann/pytorch_prototyping/blob/master/pytorch_prototyping.py
    """
    def __init__(self, in_dim, out_dim, with_ln=True):
        super().__init__()
        self.net = [nn.Linear(in_dim, out_dim)]
        if with_ln:
            self.net += [nn.LayerNorm([out_dim])]
        self.net += [nn.ReLU()]
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        return self.net(x) 

class ImplicitField(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, num_layers, 
                outmost_linear=False, with_ln=True, skips=[], spec_init=True):
        super().__init__()
        self.skips = skips
        self.net = []

        prev_dim = in_dim
        for i in range(num_layers):
            next_dim = out_dim if i == (num_layers - 1) else hidden_dim
            if (i == (num_layers - 1)) and outmost_linear:
                self.net.append(nn.Linear(prev_dim, next_dim))
            else:
                self.net.append(FCLayer(prev_dim, next_dim, with_ln=with_ln))
            prev_dim = next_dim
            if (i in self.skips) and (i != (num_layers - 1)):
                prev_dim += in_dim
        
        if num_layers > 0:
            self.net = nn.ModuleList(self.net)
            if spec_init:
                self.net.apply(self.init_weights)

    def forward(self, x):
        y = x
        for i, layer in enumerate(self.net):
            if i-1 in self.skips:
                y = torch.cat((x, y), dim=-1)
            y = layer(y)
        return y

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')


class NSVF_MLP(nn.Module):
    def __init__(self, 
        in_ch_pts: int=416, in_ch_dir: int=24,
        D_feat: int=3, W_feat: int=256, skips_feat: List[int]=[],
        D_sigma: int=2, W_sigma: int=128, skips_sigma: List[int]=[],
        D_rgb: int=5, W_rgb: int=256, skips_rgb: List[int]=[],
        layernorm: bool=True, with_activation: bool=False, 
        with_n2_term: bool=True 
        ):
        '''
        Re-organize NSVF's MLP definition into one simple nn.Module, easier for users to see 
        the whole NN architecture and to make their own modifications.
        NSVF's core MLP is actually very similar to NeRF's MLP
        '''
        super().__init__()
        self.with_n2_term = with_n2_term
        self.feat_layers = \
            ImplicitField(in_ch_pts, W_feat, W_feat, D_feat, with_ln=layernorm, skips=skips_feat)

        self.sigma_layers = nn.Sequential(
            ImplicitField(W_feat, 1, W_sigma, D_sigma, 
                with_ln=layernorm, skips=skips_sigma, outmost_linear=True),
            nn.ReLU(True) if with_activation else nn.Identity()
        )

        out_ch_rgb = 3 # or 4 if with_alpha
        self.rgb_layers = nn.Sequential(
            ImplicitField(W_feat+in_ch_dir, out_ch_rgb, W_rgb, D_rgb,
                with_ln=layernorm, skips=skips_rgb, outmost_linear=True),
            nn.Sigmoid() if with_activation else nn.Identity()
        )
    
    def forward(self, pts: Tensor, dir: Optional[Tensor]=None):
        feat = self.feat_layers(pts)
        sigma = self.sigma_layers(feat)
        ret = {'sigma': sigma}
        if self.with_n2_term:
            ret['feat_n2'] = (feat ** 2).sum(-1)
        if dir is None:
            return ret
        rgb_input = torch.cat([feat, dir], -1)
        rgb = self.rgb_layers(rgb_input)
        ret['rgb'] = rgb
        return ret

class BackgroundField(nn.Module):
    """
    Background (we assume a uniform color)
    """
    def __init__(self, out_dim=3, bg_color=1.0, min_color=-1, 
        trainable=True, background_depth=5.0):
        '''
        Args:
            bg_color: List or Scalar
            min_color: int
        '''
        super().__init__()

        if out_dim == 3:  # directly model RGB
            if isinstance(bg_color, List) or isinstance(bg_color, Tuple):
                assert len(bg_color) in [3,1], "bg_color should have size 1 or 3"
            else:
                bg_color = [bg_color]
            if min_color == -1:
                bg_color = [b * 2 - 1 for b in bg_color]
            if len(bg_color) == 1:
                bg_color = bg_color + bg_color + bg_color
            bg_color = torch.tensor(bg_color)
        else:    
            bg_color = torch.ones(out_dim).uniform_()
            if min_color == -1:
                bg_color = bg_color * 2 - 1
        self.out_dim = out_dim
        self.bg_color = nn.Parameter(bg_color, requires_grad=trainable)
        self.depth = background_depth

    def forward(self, x, **kwargs):
        return self.bg_color.unsqueeze(0).expand(
            *x.size()[:-1], self.out_dim)