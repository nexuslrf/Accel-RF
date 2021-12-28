import unittest
import os, sys
import logging
logging.basicConfig(level=logging.INFO)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from accelRF.datasets import Blender
from accelRF.raysampler import NeRFRaySampler


class TestNeRFRaySampler(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName=methodName)
        self.dataset = Blender('/data/stu01/ruofan/nerf-pytorch/data/nerf_synthetic/', 'lego')
        self.batch_shape = torch.Size([2048,3])

    @unittest.SkipTest
    def test_no_batching(self):
        raysampler = NeRFRaySampler(self.dataset)
        logging.info('no batching')
        logging.info(f'raysampler: {raysampler.coords.shape}')
        out = raysampler[0]
        self.assertEqual(out['rays_o'].shape, self.batch_shape)
        self.assertEqual(out['rays_d'].shape, self.batch_shape)
        self.assertEqual(out['gt_rgb'].shape, self.batch_shape)
    
    @unittest.SkipTest
    def test_use_batching(self):
        raysampler = NeRFRaySampler(self.dataset, use_batching=True)
        logging.info('use batching')
        logging.info(f'rays_cat: {raysampler.rays_rgb.shape}')
        out = raysampler[0]
        self.assertEqual(out['rays_o'].shape, self.batch_shape)
        self.assertEqual(out['rays_d'].shape, self.batch_shape)
        self.assertEqual(out['gt_rgb'].shape, self.batch_shape)

    # test the functionality on GPU
    @unittest.SkipTest
    def test_no_batching_cuda(self):
        raysampler = NeRFRaySampler(self.dataset.to('cuda:0'), device='cuda:0')
        logging.info('no batching')
        logging.info(f'raysampler: {raysampler.coords.shape}')
        out = raysampler[0]
        logging.info('device: '+ out['rays_o'].device)
        self.assertEqual(out['rays_o'].shape, self.batch_shape)
        self.assertEqual(out['rays_d'].shape, self.batch_shape)
        self.assertEqual(out['gt_rgb'].shape, self.batch_shape)

    @unittest.SkipTest
    def test_use_batching_cuda(self):
        raysampler = NeRFRaySampler(self.dataset.to('cuda:0'), use_batching=True, device='cuda:0')
        logging.info('use batching')
        logging.info(f'rays_cat: {raysampler.rays_rgb.shape}')
        out = raysampler[0]
        logging.info('device: '+ out['rays_o'].device)
        self.assertEqual(out['rays_o'].shape, self.batch_shape)
        self.assertEqual(out['rays_d'].shape, self.batch_shape)
        self.assertEqual(out['gt_rgb'].shape, self.batch_shape)

    @unittest.SkipTest
    def test_attribute_update_with_dataloader(self):
        """To see difference, the __getitem__ function's return is modified into:
        gt_rgb has different value when precrop is True / False
        """
        for i in range(3):
            raysampler = NeRFRaySampler(self.dataset, length=4)
            logging.info(f'{i} workers')
            rayloader = DataLoader(raysampler, 
                    batch_size=1, shuffle=True, num_workers=i, pin_memory=True)
            for batch in rayloader:
                logging.info('batch: ' + batch['gt_rgb'].sum())
            raysampler.disable_precrop()
            logging.info('After disabling in raysampler...')
            for batch in rayloader:
                logging.info('batch: ' + batch['gt_rgb'].sum())
            rayloader.dataset.disable_precrop()
            logging.info('After disabling in loader.dataset...')
            for batch in rayloader:
                logging.info('batch: ' + batch['gt_rgb'].sum())

        logging.info('Mode II')
        for i in range(3):
            raysampler = NeRFRaySampler(self.dataset, length=4)
            logging.info(f'{i} workers')
            rayloader = DataLoader(raysampler, 
                    batch_size=1, shuffle=True, num_workers=i, pin_memory=True)
            cnt = 0
            while cnt < 12:
                for batch in rayloader:
                    logging.info('batch: ' + batch['gt_rgb'].sum())
                    cnt+=1
                    if cnt == 6:
                        raysampler.disable_precrop()
                        logging.info('After disabling in raysampler...')
                    if cnt == 10:
                        rayloader.dataset.disable_precrop()
                        logging.info('After disabling in loader.dataset...')

    def test_full_rendering(self):
        raysampler = NeRFRaySampler(self.dataset, full_rendering=True, precrop=False)
        logging.info('rendering full image')
        tar_shape = torch.Size([800*800, 3])
        out = raysampler[0]
        self.assertEqual(out['rays_o'].shape, tar_shape)
        self.assertEqual(out['rays_d'].shape, tar_shape)
        self.assertEqual(out['gt_rgb'].shape, tar_shape)

if __name__ == '__main__':
    unittest.main(exit=False)