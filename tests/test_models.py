import unittest
import os, sys
import logging
logging.basicConfig(level=logging.INFO)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import OrderedDict
import torch
import torch.nn as nn

import accelRF.models.modules as modules
from accelRF.models.nerf import NeRF

@unittest.SkipTest
class TestModules(unittest.TestCase):
    def test_positional_encoding(self):
        pe = modules.PositionalEncoding(N_freqs=4)
        pe = nn.DataParallel(pe).to('cuda')
        x = torch.rand(1024, 3).to('cuda')
        tar_shape = torch.Size([1024, 3+3*2*4])
        y = pe(x)
        self.assertEqual(y.shape, tar_shape)
        self.assertEqual(pe.state_dict(), OrderedDict())

class TestNeRF(unittest.TestCase):
    def test_nerf_model(self):
        model = NeRF(D=8, W=256, in_channels_pts=63, in_channels_dir=27, skips=[4]).to('cuda')
        pts = torch.randn(1024, 63).to('cuda')
        dir = torch.randn(1024, 27).to('cuda')
        sigma_shape = torch.Size([1024, 1])
        rgb_shape = torch.Size([1024, 3])
        out_sigma = model(pts)['sigma']
        self.assertEqual(out_sigma.shape, sigma_shape)
        out = model(pts, dir)
        out_sigma, out_rgb = out['sigma'], out['rgb']
        # out_sigma, out_rgb = torch.split(model(pts, dir), [1, 3], -1)
        self.assertEqual(out_sigma.shape, sigma_shape)
        self.assertEqual(out_rgb.shape, rgb_shape)

    def test_nerf_model_jit(self):
        model = NeRF(D=8, W=256, in_channels_pts=63, in_channels_dir=27, skips=[4]).to('cuda')
        pts = torch.randn(1024, 63).to('cuda')
        dir = torch.randn(1024, 27).to('cuda')
        # out_sigma, out_rgb = torch.split(model(pts, dir), [1, 3], -1)
        out = model(pts, dir)
        out_sigma, out_rgb = out['sigma'], out['rgb']
        # jit
        model_jit = torch.jit.script(model)
        # out_sigma_, out_rgb_ = torch.split(model_jit(pts, dir), [1, 3], -1)
        self.assertEqual((out_sigma - model_jit(pts)['sigma']).norm(), 0)
        out_ = model_jit(pts, dir)
        out_sigma_, out_rgb_ = out_['sigma'], out_['rgb']
        self.assertEqual((out_sigma - out_sigma_).norm(), 0)
        self.assertEqual((out_rgb - out_rgb_).norm(), 0)
        # backward
        loss_ = (out_rgb_ ** 2).mean()
        loss_.backward()
        grad_ = model_jit.pts_feature_layer.weight.grad.sum()
        logging.info(grad_)
        logging.info(model.pts_feature_layer.weight.grad.sum())
        model_jit.zero_grad()
        self.assertEqual(model_jit.pts_feature_layer.weight.grad.sum(), 0)
        
        loss = (out_rgb ** 2).mean()
        loss.backward()
        grad = model.pts_feature_layer.weight.grad.sum()
        model.zero_grad()
        self.assertEqual(grad, grad_)

if __name__ == '__main__':
    unittest.main(exit=False)