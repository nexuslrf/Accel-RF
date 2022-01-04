import unittest
import os, sys
import logging

logging.basicConfig(level=logging.INFO)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch.utils.data import DataLoader

from accelRF.datasets import Blender, LLFF

@unittest.SkipTest
class TestBlenderDataset(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName=methodName)
        self.dataset = Blender('/data/stu01/ruofan/nerf-experiment/data/nerf_synthetic/', 'lego')
        H,W,f = self.dataset.get_hwf()
        self.img_shape = torch.Size([H,W,3])
        self.pose_shape = torch.Size([3,4])

    def test_subsets(self):
        len_dataset = len(self.dataset)
        train_set = self.dataset.get_sub_set('train')
        val_set = self.dataset.get_sub_set('val')
        test_set = self.dataset.get_sub_set('test')
        n_frame = 20
        render_set = self.dataset.get_render_set(n_frame=n_frame)
        self.assertEqual(len(train_set), len(self.dataset.i_split['train']))
        self.assertEqual(len(val_set), len(self.dataset.i_split['val']))
        self.assertEqual(len(test_set), len(self.dataset.i_split['test']))
        self.assertEqual(len(render_set), n_frame)
        self.assertEqual(len(self.dataset), len_dataset)
    
    @unittest.SkipTest
    def test_dataloader_on_dataset(self):
        BS = 8
        train_loader = DataLoader(self.dataset.get_sub_set('train'), batch_size=BS, shuffle=True, num_workers=0, pin_memory=True)
        render_loader = DataLoader(self.dataset.get_render_set(), batch_size=BS, shuffle=True, num_workers=0, pin_memory=True)
        train_batch = next(iter(train_loader))
        self.assertEqual(train_batch['pose'].shape, torch.Size([BS,*self.pose_shape]))
        self.assertEqual(train_batch['gt_img'].shape, torch.Size([BS,*self.img_shape]))
        render_batch = next(iter(render_loader))
        self.assertEqual(render_batch['pose'].shape, torch.Size([BS,*self.pose_shape]))

# @unittest.SkipTest
class TestLLFFDataset(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName=methodName)
        self.root, self.scene = '/data/stu01/ruofan/nerf-experiment/data/nerf_llff_data/', 'fern'
        self.dataset = LLFF(self.root, self.scene)
        H,W,f = self.dataset.get_hwf()
        self.img_shape = torch.Size([H,W,3])
        self.pose_shape = torch.Size([3,4])

    @unittest.SkipTest
    def test_shapes(self):
        item = self.dataset[0]
        self.assertEqual(item['pose'].shape, self.pose_shape)
        self.assertEqual(item['gt_img'].shape, self.img_shape)
    
    @unittest.SkipTest
    def test_subsets(self):
        len_dataset = len(self.dataset)
        train_set = self.dataset.get_sub_set('train')
        val_set = self.dataset.get_sub_set('val')
        test_set = self.dataset.get_sub_set('test')
        n_frame = 100
        render_set = self.dataset.get_render_set(n_frame=n_frame)
        self.assertEqual(len(train_set), len(self.dataset.i_split['train']))
        self.assertEqual(len(val_set), len(self.dataset.i_split['val']))
        self.assertEqual(len(test_set), len(self.dataset.i_split['test']))
        self.assertEqual(len(render_set), n_frame)
        self.assertEqual(len(self.dataset), len_dataset)
    
    @unittest.SkipTest
    def test_dataloader_on_dataset(self):
        BS = 8
        train_loader = DataLoader(self.dataset.get_sub_set('train'), batch_size=BS, shuffle=True, num_workers=0, pin_memory=True)
        train_batch = next(iter(train_loader))
        self.assertEqual(train_batch['pose'].shape, torch.Size([BS,*self.pose_shape]))
        self.assertEqual(train_batch['gt_img'].shape, torch.Size([BS,*self.img_shape]))
    
    def test_non_default_options(self):
        dataset = LLFF(self.root, self.scene, spherify=False, n_holdout=8)
        item = dataset[0]
        self.assertEqual(item['pose'].shape, self.pose_shape)
        val_set = dataset.get_sub_set('val')
        logging.info(f'val_len: {len(val_set)}')
        render_set = dataset.get_render_set()
        self.assertEqual(render_set[0]['pose'].shape, self.pose_shape)

if __name__ == '__main__':
    unittest.main(exit=False)
    # import copy
    # dataset = LLFF('/data/stu01/ruofan/nerf-experiment/data/nerf_llff_data/', 'fern')
    # dd = copy.copy(dataset)