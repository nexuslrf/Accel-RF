import os
import torch
import numpy as np
import cv2
import imageio
from glob import glob
from .base import BaseDataset
'''
Codes from https://github.com/lioryariv/volsdf/blob/main/code/datasets/scene_dataset.py
Not a ideal, generalizable impl., it can be improved.
'''

def glob_imgs(path):
    imgs = []
    for ext in ['*.png', '*.jpg', '*.JPEG', '*.JPG']:
        imgs.extend(glob(os.path.join(path, ext)))
    return imgs

def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K/K[2,2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3,3] = (t[:3] / t[3])[:,0]

    return intrinsics, pose


class SceneDataset(BaseDataset):

    def __init__(self,
                 data_dir,
                 scan_id=0,
                 testskip=1,
                 ):
        super().__init__()
        self.GL_coord = False

        self.instance_dir = os.path.join(data_dir, 'scan{0}'.format(scan_id))

        assert os.path.exists(self.instance_dir), "Data directory is empty"

        self.sampling_idx = None

        image_dir = '{0}/image'.format(self.instance_dir)
        image_paths = sorted(glob_imgs(image_dir))
        self.n_images = len(image_paths)

        self.cam_file = '{0}/cameras.npz'.format(self.instance_dir)
        camera_dict = np.load(self.cam_file)
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        self.intrinsics = []
        self.poses = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)
            self.intrinsics.append(torch.from_numpy(intrinsics).float())
            self.poses.append(torch.from_numpy(pose).float())
        self.poses = torch.stack(self.poses, 0)
        self.intrinsics = torch.stack(self.intrinsics, 0)
        self.focal = self.intrinsics[:, 0,0].mean().item() # just assume focal_x = focal_y...

        self.imgs = []
        for path in image_paths:
            self.imgs.append(imageio.imread(path))
        self.imgs = (np.array(self.imgs) / 255.).astype(np.float32)
        self.imgs = torch.from_numpy(self.imgs)

        self.img_shape = self.imgs[0].shape[:2]
        self.H, self.W = self.img_shape

        self.i_split = {
            'train': np.arange(self.n_images),
            'val': np.arange(0, self.n_images, testskip),
            'test': np.arange(0, self.n_images, testskip)
        }