import os
import torch
import numpy as np
import cv2
import imageio
from glob import glob
from .base import BaseDataset


class NSVFDataset(BaseDataset):

    def __init__(self,
                 data_dir,
                 scan,
                 testskip=1,
                 ):
        super().__init__()
        self.openGL_coord = False

        self.instance_dir = os.path.join(data_dir, '{0}'.format(scan))

        assert os.path.exists(self.instance_dir), f"Data directory {self.instance_dir} is empty"

        self.sampling_idx = None

        image_paths = sorted(os.listdir('{0}/rgb'.format(self.instance_dir)))
        pose_paths = sorted(os.listdir('{0}/pose'.format(self.instance_dir)))
        self.img_paths = image_paths
        self.n_images = len(image_paths)

        self.Ks = np.loadtxt(f'{self.instance_dir}/intrinsics.txt')[None,...] # [1,4,4]
        self.Ks = torch.from_numpy(self.Ks).float()
        if os.path.exists(f'{self.instance_dir}/scale_mat.txt'):
            scale_mat = np.loadtxt(f'{self.instance_dir}/scale_mat.txt') # [4,4]
            print('load scale_mat')
        else:
            scale_mat = np.eye(4)
    
        scale_t = scale_mat[:3,3]
        scale_r = np.linalg.inv(scale_mat[:3,:3])

        self.Ts = []
        for p_path in pose_paths:
            T = np.loadtxt(f'{self.instance_dir}/pose/{p_path}') # [4,4]
            T[:3,3] = (T[:3,3] - scale_t) @ scale_r
            self.Ts.append(torch.from_numpy(T).float())
        self.Ts = torch.stack(self.Ts, 0)

        self.focal = self.Ks[:, 0,0].mean().item() # just assume focal_x = focal_y...

        self.imgs = []
        for path in image_paths:
            self.imgs.append(imageio.imread(f'{self.instance_dir}/rgb/{path}'))
        self.imgs = (np.array(self.imgs) / 255.).astype(np.float32)
        self.imgs = torch.from_numpy(self.imgs)

        self.img_shape = self.imgs[0].shape[:2]
        self.H, self.W = self.img_shape

        train_split = np.array([i for i in range(len(image_paths)) if image_paths[i].startswith('0_')])
        test_split = np.array([i for i in range(len(image_paths)) if image_paths[i].startswith('1_')])
        val_split = test_split
        self.i_split = {
            'train': train_split,
            'val': test_split,
            'test': val_split
        }