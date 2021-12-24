import unittest
import os, sys
import logging
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader

from accelRF.datasets import Blender
from accelRF.raysampler import NeRFSampler


class TestNeRFSampler(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName=methodName)
        self.dataset = Blender('/data/stu01/ruofan/nerf-pytorch/data/nerf_synthetic/', 'lego')

    def test_no_batching(self):
        raysampler = NeRFSampler(self.dataset)
        log.debug('no batching')
    
    def test_use_batching(self):
        raysampler = NeRFSampler(self.dataset, use_batching=True)
        log.debug('use batching')

if __name__ == '__main__':
    unittest.main(exit=False)
