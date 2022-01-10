import unittest
import os, sys
import logging
import time
logging.basicConfig(level=logging.INFO)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from accelRF.rep.voxel_grid import VoxelGrid
import accelRF.rep.utils as utils

def bbox2voxels(bbox, voxel_size):
    vox_min, vox_max = bbox[:3], bbox[3:]
    steps = ((vox_max - vox_min) / voxel_size).round().astype('int64') + 1 # add 1 lol
    x, y, z = [c.reshape(-1).astype('float32') for c in np.meshgrid(np.arange(steps[0]), np.arange(steps[1]), np.arange(steps[2]))]
    x, y, z = x * voxel_size + vox_min[0], y * voxel_size + vox_min[1], z * voxel_size + vox_min[2]
    
    return np.stack([x, y, z]).T.astype('float32')

class TestVoxelGrid(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName=methodName)
        self.bbox_np = np.array([-0.67, -1.2, -0.37, 0.67, 1.2, 1.03])
        self.bbox_pt = torch.tensor([-0.67, -1.2, -0.37, 0.67, 1.2, 1.03])

    @unittest.SkipTest
    def test_voxels_xyz(self):
        vox_np = bbox2voxels(self.bbox_np, 0.4)
        logging.info(vox_np.shape)
        vox_pt = utils.bbox2voxels(self.bbox_pt, 0.4)
        logging.info(vox_pt.shape)

    def test_ray_intersect(self):
        from accelRF.raysampler.utils import get_rays
        import accelRF._C.rep._ext as rep_ext
        pose = torch.tensor([
            [-9.9990e-01,  4.1922e-03, -1.3346e-02, -5.3798e-02],
            [-1.3989e-02, -2.9966e-01,  9.5394e-01,  3.8455e+00],
            [-4.6566e-10,  9.5404e-01,  2.9969e-01,  1.2081e+00],
            [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]])
        H, W = 800, 800
        f = 1111.1110311937682
        voxelsize = 0.4
        rays_o, rays_d = get_rays(H, W, f, pose) # 800,800,3
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
        vox_grid = VoxelGrid(self.bbox_pt, voxelsize)
        pts = vox_grid.center_points.reshape(-1, 3) # 
        G = 1024
        N = rays_o.shape[0]
        K = (N-1)//G + 1 # 625
        rays_o = rays_o.reshape(G, K, 3).contiguous().cuda()
        rays_d = rays_d.reshape(G, K, 3).contiguous().cuda()
        pts = pts[None,:].expand(G, *pts.shape).contiguous().cuda()
        n_max = 60
        inds, min_depth, max_depth = rep_ext.aabb_intersect(
            rays_o, rays_d, pts, voxelsize, n_max)
        print(inds.shape)
        print(max_depth.shape)
        print(min_depth.shape)


if __name__ == '__main__':
    unittest.main(exit=False)