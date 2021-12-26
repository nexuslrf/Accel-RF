import unittest
import os, sys
import logging
logging.basicConfig(level=logging.INFO)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
# torch._C._jit_set_profiling_executor(False)
import accelRF.pointsampler as aps


class TestPointRaySampler(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName=methodName)
        self.N_rays = 128
        self.N_samples = 64
        self.tar_shape = torch.Size([self.N_rays, self.N_samples, 3])
        self.rays_o = torch.randn(self.N_rays, 3)
        self.rays_d = torch.randn(self.N_rays, 3)

    @unittest.SkipTest
    def test_uniform_sample_with_init_z(self):
        # CPU version
        init_z_vals = torch.linspace(0, 1, steps=self.N_samples)
        pts = aps.uniform_sample(self.N_samples, 1., 6., self.rays_o, self.rays_d, init_z_vals=init_z_vals)
        self.assertEqual(pts.shape, self.tar_shape)

    @unittest.SkipTest
    def test_uniform_sample_without_init_z(self):
        # CPU version
        pts = aps.uniform_sample(self.N_samples, 1., 6., self.rays_o, self.rays_d)
        self.assertEqual(pts.shape, self.tar_shape)

    # @unittest.SkipTest
    def test_uniform_sample_with_init_z_cuda(self):
        # GPU version
        init_z_vals = torch.linspace(0, 1, steps=self.N_samples).cuda()
        pts = aps.uniform_sample(self.N_samples, 1., 6., self.rays_o.cuda(), self.rays_d.cuda(), init_z_vals=init_z_vals)
        self.assertEqual(pts.shape, self.tar_shape)

    @unittest.SkipTest
    def test_uniform_sample_without_init_z_cuda(self):
        # GPU version
        pts = aps.uniform_sample(self.N_samples, 1., 6., self.rays_o.cuda(), self.rays_d.cuda())
        self.assertEqual(pts.shape, self.tar_shape)

    @unittest.SkipTest
    def test_uniform_sample_with_perturb(self):
        pts = aps.uniform_sample(self.N_samples, 1., 6., self.rays_o, self.rays_d, perturb=1.)
        self.assertEqual(pts.shape, self.tar_shape)

    @unittest.SkipTest
    def test_uniform_sample_jit(self):
        logging.info(aps.uniform_sample.code)

if __name__ == '__main__':
    unittest.main(exit=False)
