import unittest
import os, sys
import logging
import time
logging.basicConfig(level=logging.INFO)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from accelRF.render.nerf_render import volumetric_rendering

class TestNeRFRendering(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName=methodName)
        self.N_rays = 1024
        self.N_samples = 128
    
    # @unittest.SkipTest
    def test_vol_rendering(self):
        device = 'cuda'
        sigma = torch.relu(torch.randn(self.N_rays, self.N_samples, 1, device=device))
        rgb = torch.sigmoid(torch.randn(self.N_rays, self.N_samples, 3, device=device))
        z_vals = torch.linspace(0, 1, steps=self.N_samples, device=device)[None,:]
        z_vals_ = z_vals.expand(self.N_rays, self.N_samples)
        ray_lens = torch.randn(self.N_rays, 1, device=device)
        tar_rgb_shape = torch.Size([self.N_rays, 3])

        ret = volumetric_rendering(rgb, sigma, z_vals, ray_lens)
        self.assertEqual(ret['rgb'].shape, tar_rgb_shape)
        
        ret = volumetric_rendering(rgb, sigma, z_vals_, ray_lens)
        self.assertEqual(ret['rgb'].shape, tar_rgb_shape)

        # rgb.requires_grad = True
        # ret = volumetric_rendering(rgb, sigma, z_vals_, ray_lens)
        # loss = (ret['rgb']**2).mean()
        # loss.backward()
        # logging.info(rgb.grad.sum())

        # speedtest
        logging.info('Non JIT script')
        for i in range(10):
            t0 = time.time()
            ret = volumetric_rendering(rgb, sigma, z_vals, ray_lens)
            f'{ret["rgb"][0]}'
            logging.info(f'Round: {i}, Time: {(time.time()-t0)*1000} ms')
        logging.info('JIT script')
        volumetric_rendering_jit = torch.jit.script(volumetric_rendering)
        for i in range(10):
            t0 = time.time()
            ret = volumetric_rendering_jit(rgb, sigma, z_vals, ray_lens)
            f'{ret["rgb"][0]}'
            logging.info(f'Round: {i}, Time: {(time.time()-t0)*1000} ms')

if __name__ == '__main__':
    unittest.main(exit=False)