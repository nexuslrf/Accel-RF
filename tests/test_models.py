import unittest
import os, sys
import logging
logging.basicConfig(level=logging.INFO)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import OrderedDict
import torch
import torch.nn as nn

import accelRF.models.encodings as encodings
from accelRF.models.nerf import NeRF
from accelRF.models.nsvf import NSVF_MLP
from accelRF.rep.voxel_grid import VoxelGrid

@unittest.SkipTest
class TestModules(unittest.TestCase):
    def test_positional_encoding(self):
        pe = encodings.PositionalEncoding(N_freqs=4)
        pe = torch.jit.script(pe)
        pe = nn.DataParallel(pe).to('cuda')
        pe = pe.to('cuda')
        x = torch.rand(1024, 3).to('cuda')
        y = pe(x)
        self.assertEqual(y.shape, torch.Size([1024, 3+3*2*4]))
        x2 = torch.rand(16, 64, 3).to('cuda')
        y2 = pe(x2)
        self.assertEqual(y2.shape, torch.Size([16, 64, 3+3*2*4]))
        # self.assertEqual(pe.state_dict(), OrderedDict())

@unittest.SkipTest
class TestNeRF(unittest.TestCase):
    def test_nerf_model(self):
        model = NeRF(D=8, W=256, in_channels_pts=63, in_channels_dir=27, skips=[4]).to('cuda')
        pts = torch.randn(16, 64, 63).to('cuda')
        dir = torch.randn(16, 64, 27).to('cuda')
        sigma_shape = torch.Size([16, 64, 1])
        rgb_shape = torch.Size([16, 64, 3])
        out_sigma = model(pts)['sigma']
        self.assertEqual(out_sigma.shape, sigma_shape)
        out = model(pts, dir)
        out_sigma, out_rgb = out['sigma'], out['rgb']
        # out_sigma, out_rgb = torch.split(model(pts, dir), [1, 3], -1)
        self.assertEqual(out_sigma.shape, sigma_shape)
        self.assertEqual(out_rgb.shape, rgb_shape)

    @unittest.SkipTest
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

class TestVoxEncoding(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName=methodName)
        bbox = torch.tensor([-0.67, -1.2, -0.37, 0.67, 1.2, 1.03])
        voxelsize = 0.4
        self.vox_grid = VoxelGrid(bbox, voxelsize).to('cuda')

    def test_voxel_cdf_sample(self):
        from accelRF.raysampler.utils import get_rays
        import accelRF.pointsampler as aps
        pose = torch.tensor([
            [-9.9990e-01,  4.1922e-03, -1.3346e-02, -5.3798e-02],
            [-1.3989e-02, -2.9966e-01,  9.5394e-01,  3.8455e+00],
            [-4.6566e-10,  9.5404e-01,  2.9969e-01,  1.2081e+00],
            [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]])
        H, W = 800, 800
        f = 1111.1110311937682
        rays_o, rays_d = get_rays(H, W, f, pose, normalize_dir=True) # 800,800,3
        rays_o = rays_o.reshape(-1, 3).contiguous().cuda()
        rays_d = rays_d.reshape(-1, 3).contiguous().cuda()
        vox_idx, t_near, t_far, hits = self.vox_grid.ray_intersect(rays_o, rays_d)
        N_sample = 1024
        vox_idx = vox_idx[:N_sample]
        t_near, t_far = t_near[:N_sample], t_far[:N_sample]
        rays_o, rays_d = rays_o[:N_sample], rays_d[:N_sample]

        pts, p2v_idx, t_vals = aps.voxel_cdf_sample(
            rays_o, rays_d, vox_idx, t_near, t_far, 0.125)
        pts_mask = p2v_idx.ne(-1)

        VoxEncoding = encodings.VoxelEncoding(self.vox_grid.n_corners, 16).cuda()

        pts_in, p2v_idx_in = pts[pts_mask], p2v_idx[pts_mask]
        logging.info(f'{pts_in.shape} {p2v_idx_in.shape}')
        embeds = VoxEncoding(pts_in, p2v_idx_in)
        logging.info(embeds.shape)
        self.assertEqual(embeds.shape, torch.Size([pts_in.shape[0], 16]))

if __name__ == '__main__':
    unittest.main(exit=False)