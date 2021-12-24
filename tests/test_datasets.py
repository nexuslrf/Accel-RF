import unittest
import os, sys
import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader

from accelRF.datasets import Blender

class TestDataset(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName=methodName)
        self.dataset = Blender('/data/stu01/ruofan/nerf-pytorch/data/nerf_synthetic/', 'lego')
    
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
    
    def test_dataloader_on_dataset(self):
        train_loader = DataLoader(self.dataset.get_sub_set('train'), batch_size=8, shuffle=True, num_workers=0, pin_memory=True)
        render_loader = DataLoader(self.dataset.get_render_set(), batch_size=8, shuffle=True, num_workers=0, pin_memory=True)
        train_batch = next(iter(train_loader))
        log.info('dataloader_on_dataset')
        log.info(train_batch['pose'].shape, train_batch['gt_img'].shape)
        render_batch = next(iter(render_loader))
        log.info(render_batch['pose'].shape)


if __name__ == '__main__':
    unittest.main(exit=False)
