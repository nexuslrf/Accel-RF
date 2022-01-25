import unittest
import os, sys
import logging
logging.basicConfig(level=logging.INFO)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
# torch._C._jit_set_profiling_executor(False)
import accelRF.pointsampler as aps
from accelRF.rep.voxel_grid import VoxelGrid


class TestNeRFPointSampler(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName=methodName)
        self.N_rays = 128
        self.N_samples = 64
        self.N_importance = 128
        self.tar_shape = torch.Size([self.N_rays, self.N_samples, 3])
        self.rays_o = torch.randn(self.N_rays, 3)
        self.rays_d = torch.randn(self.N_rays, 3)

    @unittest.SkipTest
    def test_uniform_sample_with_init_z(self):
        # CPU version
        init_z_vals = torch.linspace(0, 1, steps=self.N_samples)
        pts, _ = aps.uniform_sample(self.N_samples, self.rays_o, self.rays_d, 1., 6., init_z_vals=init_z_vals)
        self.assertEqual(pts.shape, self.tar_shape)

    @unittest.SkipTest
    def test_uniform_sample_without_init_z(self):
        # CPU version
        pts, _ = aps.uniform_sample(self.N_samples, self.rays_o, self.rays_d, 1., 6.)
        self.assertEqual(pts.shape, self.tar_shape)

    @unittest.SkipTest
    def test_uniform_sample_with_init_z_cuda(self):
        # GPU version
        init_z_vals = torch.linspace(0, 1, steps=self.N_samples).cuda()
        pts, _ = aps.uniform_sample(self.N_samples, self.rays_o.cuda(), self.rays_d.cuda(), 1., 6., init_z_vals=init_z_vals)
        self.assertEqual(pts.shape, self.tar_shape)

    # @unittest.SkipTest
    def test_uniform_sample_without_init_z_cuda(self):
        # GPU version
        pts, _ = aps.uniform_sample(self.N_samples, self.rays_o.cuda(), self.rays_d.cuda(), 1., 6.)
        self.assertEqual(pts.shape, self.tar_shape)
    
    # @unittest.SkipTest
    def test_uniform_sample_z_only_cuda(self):
        # GPU version
        pts, z_vals = aps.uniform_sample(self.N_samples, self.rays_o.cuda(), self.rays_d.cuda(), 1., 6., only_z_vals=True)
        logging.info(z_vals.shape)
        init_z_vals =aps.get_z_vals(0, torch.ones(10,1), 20)
        logging.info(init_z_vals.shape)
        # pts, z_vals = aps.uniform_sample(
        #     self.N_samples, 1., torch.ones(self.N_rays,1).cuda(), self.rays_o.cuda(), self.rays_d.cuda())

    @unittest.SkipTest
    def test_uniform_sample_with_perturb(self):
        pts, _ = aps.uniform_sample(self.N_samples, self.rays_o, self.rays_d, 1., 6., perturb=1.)
        self.assertEqual(pts.shape, self.tar_shape)

    @unittest.SkipTest
    def test_uniform_sample_jit(self):
        logging.info(aps.uniform_sample.code)


    @unittest.SkipTest
    def test_sample_pdf(self):
        z_vals = torch.linspace(0, 1, steps=self.N_samples)[None,:]#.cuda()
        weights = torch.rand(self.N_rays, z_vals.shape[-1])#.cuda()
        samples = aps.sample_pdf(z_vals, weights[...,1:], self.N_importance)
        tar_shape = torch.Size([self.N_rays, self.N_importance])
        self.assertEqual(samples.shape, tar_shape)

    @unittest.SkipTest
    def test_sample_pdf_no_grad(self):
        z_vals = torch.linspace(0, 1, steps=self.N_samples)[None,:]#.cuda()
        weights = torch.rand(self.N_rays, z_vals.shape[-1])#.cuda()
        weights.requires_grad = True
        # loss = (weights **2).sum() + weights.sum()
        # loss.backward()
        # logging.info(weights.grad)
        samples = aps.sample_pdf(z_vals, weights[...,1:], self.N_importance)
        loss = (samples**2).sum() + weights.sum()
        loss.backward()
        expected_grad = torch.ones_like(weights)
        self.assertEqual((weights.grad - expected_grad).norm(), 0)
    
    @unittest.SkipTest
    def test_cdf_sample(self):
        z_vals = torch.linspace(0, 1, steps=self.N_samples)[None,:]#.cuda()
        weights = torch.rand(self.N_rays, z_vals.shape[-1])#.cuda()
        pts, _ = aps.cdf_sample(self.N_importance, self.rays_o, self.rays_d, z_vals, weights, 0)
        self.assertEqual(pts.shape, torch.Size([self.N_rays, self.N_samples+self.N_importance, 3]))
        # logging.info(aps.cdf_sample.graph)
    
    @unittest.SkipTest
    def test_wrapper(self):
        device = 'cuda'
        sampler = aps.NeRFPointSampler(self.N_samples, 1., 6., self.N_importance)#.to(device)
        sampler = nn.DataParallel(sampler).to(device)
        z_vals = torch.linspace(0, 1, steps=self.N_samples)[None,:].to(device)
        weights = torch.rand(self.N_rays, z_vals.shape[-1]).to(device)
        ret = sampler(self.rays_o.to(device), self.rays_d.to(device))
        pts_coarse, _ = ret # must be in this way if using DataParallel
        self.assertEqual(pts_coarse.shape, self.tar_shape)
        ret = sampler(self.rays_o.to(device), self.rays_d.to(device), z_vals, weights)
        pts_fine, _ = ret
        self.assertEqual(pts_fine.shape, torch.Size([self.N_rays, self.N_samples+self.N_importance, 3]))

@unittest.SkipTest
class TestNSVFPointSampler(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName=methodName)
        bbox = torch.tensor([-0.67, -1.2, -0.37, 0.67, 1.2, 1.03])
        voxelsize = 0.4
        self.vox_grid = VoxelGrid(bbox, voxelsize).to('cuda')

    def test_voxel_cdf_sample(self):
        from accelRF.raysampler.utils import get_rays
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
        nsvf_pts_sampler = aps.NSVFPointSampler(0.125)
        pts, p2v_idx, t_vals = nsvf_pts_sampler(rays_o, rays_d, vox_idx, t_near, t_far)
        logging.info(pts.shape)
        logging.info(t_vals.shape)
        logging.info(vox_idx.dtype)

if __name__ == '__main__':
    unittest.main(exit=False)
