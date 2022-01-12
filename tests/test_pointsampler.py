import unittest
import os, sys
import logging
logging.basicConfig(level=logging.INFO)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
# torch._C._jit_set_profiling_executor(False)
import accelRF.pointsampler as aps


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
        pts, _ = aps.uniform_sample(self.N_samples, 1., 6., self.rays_o, self.rays_d, init_z_vals=init_z_vals)
        self.assertEqual(pts.shape, self.tar_shape)

    @unittest.SkipTest
    def test_uniform_sample_without_init_z(self):
        # CPU version
        pts, _ = aps.uniform_sample(self.N_samples, 1., 6., self.rays_o, self.rays_d)
        self.assertEqual(pts.shape, self.tar_shape)

    @unittest.SkipTest
    def test_uniform_sample_with_init_z_cuda(self):
        # GPU version
        init_z_vals = torch.linspace(0, 1, steps=self.N_samples).cuda()
        pts, _ = aps.uniform_sample(self.N_samples, 1., 6., self.rays_o.cuda(), self.rays_d.cuda(), init_z_vals=init_z_vals)
        self.assertEqual(pts.shape, self.tar_shape)

    @unittest.SkipTest
    def test_uniform_sample_without_init_z_cuda(self):
        # GPU version
        pts, _ = aps.uniform_sample(self.N_samples, 1., 6., self.rays_o.cuda(), self.rays_d.cuda())
        self.assertEqual(pts.shape, self.tar_shape)

    @unittest.SkipTest
    def test_uniform_sample_with_perturb(self):
        pts, _ = aps.uniform_sample(self.N_samples, 1., 6., self.rays_o, self.rays_d, perturb=1.)
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

if __name__ == '__main__':
    unittest.main(exit=False)
