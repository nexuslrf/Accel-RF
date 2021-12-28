import unittest
import os, sys
import logging
logging.basicConfig(level=logging.INFO)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
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
    def test_coarse_sample_with_init_z(self):
        # CPU version
        init_z_vals = torch.linspace(0, 1, steps=self.N_samples)
        pts, _ = aps.coarse_sample(self.N_samples, 1., 6., self.rays_o, self.rays_d, init_z_vals=init_z_vals)
        self.assertEqual(pts.shape, self.tar_shape)

    @unittest.SkipTest
    def test_coarse_sample_without_init_z(self):
        # CPU version
        pts, _ = aps.coarse_sample(self.N_samples, 1., 6., self.rays_o, self.rays_d)
        self.assertEqual(pts.shape, self.tar_shape)

    @unittest.SkipTest
    def test_coarse_sample_with_init_z_cuda(self):
        # GPU version
        init_z_vals = torch.linspace(0, 1, steps=self.N_samples).cuda()
        pts, _ = aps.coarse_sample(self.N_samples, 1., 6., self.rays_o.cuda(), self.rays_d.cuda(), init_z_vals=init_z_vals)
        self.assertEqual(pts.shape, self.tar_shape)

    @unittest.SkipTest
    def test_coarse_sample_without_init_z_cuda(self):
        # GPU version
        pts, _ = aps.coarse_sample(self.N_samples, 1., 6., self.rays_o.cuda(), self.rays_d.cuda())
        self.assertEqual(pts.shape, self.tar_shape)

    @unittest.SkipTest
    def test_coarse_sample_with_perturb(self):
        pts, _ = aps.coarse_sample(self.N_samples, 1., 6., self.rays_o, self.rays_d, perturb=1.)
        self.assertEqual(pts.shape, self.tar_shape)

    @unittest.SkipTest
    def test_coarse_sample_jit(self):
        logging.info(aps.coarse_sample.code)


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
    def test_fine_sample(self):
        z_vals = torch.linspace(0, 1, steps=self.N_samples)[None,:]#.cuda()
        weights = torch.rand(self.N_rays, z_vals.shape[-1])#.cuda()
        pts, _ = aps.fine_sample(self.N_importance, self.rays_o, self.rays_d, z_vals, weights, 0)
        self.assertEqual(pts.shape, torch.Size([self.N_rays, self.N_samples+self.N_importance, 3]))
        # logging.info(aps.fine_sample.graph)

if __name__ == '__main__':
    unittest.main(exit=False)