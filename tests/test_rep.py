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
    
    @unittest.SkipTest
    def test_voxel_grid_class(self):
        vox_grid = VoxelGrid(self.bbox_pt, 0.4).to('cuda')
        logging.info(vox_grid.grid_shape)
        logging.info(vox_grid.center2corner.shape)
        logging.info(f'{vox_grid.center2corner[0]}, {vox_grid.center2corner[1]}')

    # @unittest.SkipTest
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
        rays_o, rays_d = get_rays(H, W, f, pose, True) # 800,800,3
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
        vox_grid = VoxelGrid(self.bbox_pt, voxelsize).to('cuda')
        pts = vox_grid.center_points # 
        n_rays = rays_o.shape[0]
        # version 1
        G = 1024
        N = rays_o.shape[0]
        K = (N-1)//G + 1 # 625
        rays_o_ = rays_o.reshape(G, K, 3).contiguous().cuda()
        rays_d_ = rays_d.reshape(G, K, 3).contiguous().cuda()
        pts_ = pts.reshape(1,-1,3).contiguous().cuda()
        n_max = 60
        inds, min_depth, max_depth = rep_ext.aabb_intersect_old(
            rays_o_, rays_d_, pts_, voxelsize, n_max)
        print(inds.shape)
        print(max_depth.shape)
        print(min_depth.shape)
        # version 2
        rays_o_ = rays_o.contiguous().cuda()
        rays_d_ = rays_d.contiguous().cuda()
        pts_ = pts.contiguous().cuda()
        inds_x, min_depth_x, max_depth_x = rep_ext.aabb_intersect(
            rays_o_, rays_d_, pts_, voxelsize, sum(pts_.shape[:3]))
        print(inds_x.shape)
        print(max_depth_x.shape)
        print(min_depth_x.shape)
        max_hit = inds_x.shape[-1]
        self.assertEqual(
            ((min_depth.reshape(-1, n_max)[:,:max_hit]-min_depth_x)**2).sum(), 0)
        # class method version
        inds_m, min_depth_m, max_depth_m, hits = vox_grid.ray_intersect(rays_o_, rays_d_)
        self.assertEqual(inds_x.sum() - inds_m.sum(), 0)
        print(inds_m.shape, hits.shape)

    @unittest.SkipTest
    def test_splitting(self):
        logging.info('Test splitting')
        bbox_pt = torch.tensor([0,0,0, 1.6, 2.4, 2.0])
        vox_grid = VoxelGrid(bbox_pt, 0.4).to('cuda')
        logging.info(vox_grid.grid_shape)
        logging.info(vox_grid.center2corner.shape)
        logging.info(f'{vox_grid.center2corner[0]}, {vox_grid.center2corner[1]}')
        corner_coords = torch.stack(torch.meshgrid([torch.arange(s) for s in vox_grid.grid_shape+1]), -1)
        corner_feats = (corner_coords * 0.4 - 0.2).reshape(-1, 3).cuda()
        offset = utils.offset_points(scale=-1, device='cuda')
        logging.info(f'corner_feats: {corner_feats.shape}')
        new_corner_feats = vox_grid.splitting(corner_feats)
        offset_ = ((new_corner_feats[vox_grid.center2corner] - vox_grid.center_points[:,None,:]) / 0.1).mean(0)
        self.assertTrue((offset-offset_).norm() < 1e-5)
        logging.info(vox_grid.grid_shape)
        logging.info(vox_grid.center2corner.shape)
        logging.info(f'{vox_grid.center2corner[0]}, {vox_grid.center2corner[1]}')

    @unittest.SkipTest
    def test_pruning(self):
        logging.info('Test pruning')
        vox_grid = VoxelGrid(self.bbox_pt, 0.4).to('cuda')
        keep = torch.rand(vox_grid.n_voxels, device=vox_grid.center_points.device) < 0.6
        logging.info(vox_grid.center2corner.shape)
        logging.info(vox_grid.n_corners)
        logging.info(f'{keep.sum()}, {keep.shape}')
        corner_idx = vox_grid.pruning(keep)
        logging.info(vox_grid.center2corner.shape)
        logging.info(vox_grid.n_corners)
        logging.info(corner_idx.shape)


if __name__ == '__main__':
    unittest.main(exit=False)