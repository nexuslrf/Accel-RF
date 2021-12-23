import os
import json
import numpy as np
import imageio
import cv2

import torch
import copy
from .base import BaseDataset

# TODO add auto data downloading?

# TODO this bounding box values are copy from KiloNeRF
bounding_box = {

}

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()

def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(
        np.array([[-1,0,0,0],
                  [ 0,0,1,0],
                  [ 0,1,0,0],
                  [ 0,0,0,1]])) @ c2w
    return c2w

class Blender(BaseDataset):
    """
    NeRF Blender datasets used in original NeRF paper.

    Args:
        -
        -
        -
    """

    def __init__(
        self, root: str, scene: str, 
        half_res: bool = False, testskip: int = 1, white_bkgd: bool = False
    )->None:
        super().__init__()
        basedir = os.path.join(root, scene)
        splits = ['train', 'val', 'test']
        metas = {}
        for s in splits:
            with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
                metas[s] = json.load(fp)

        all_imgs = []
        all_poses = []
        counts = [0]
        for s in splits:
            meta = metas[s]
            imgs = []
            poses = []
            if s=='train' or testskip==0:
                skip = 1
            else:
                skip = testskip

            for frame in meta['frames'][::skip]:
                fname = os.path.join(basedir, frame['file_path'] + '.png')
                imgs.append(imageio.imread(fname))
                poses.append(np.array(frame['transform_matrix']))
            imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
            poses = np.array(poses).astype(np.float32)
            counts.append(counts[-1] + imgs.shape[0])
            all_imgs.append(imgs)
            all_poses.append(poses)

        i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]

        imgs = np.concatenate(all_imgs, 0)
        poses = np.concatenate(all_poses, 0)

        self.img_shape = imgs[0].shape[:2]
        H, W = self.img_shape
        camera_angle_x = float(meta['camera_angle_x'])
        focal = .5 * W / np.tan(.5 * camera_angle_x)

        if half_res:
            H = H//2
            W = W//2
            focal = focal/2.

            imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
            for i, img in enumerate(imgs):
                # According to the api defined in the link below, the dimension 
                # should be represented as (W, H).
                # https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/
                imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
                # imgs_half_res[i] = cv2.resize(img, (H, W), interpolation=cv2.INTER_AREA)
            imgs = imgs_half_res

        self.near, self.far = 2., 6.
        if imgs.shape[-1] == 4:
            if white_bkgd:
                imgs = imgs[...,:3]*imgs[...,-1:] + (1.-imgs[...,-1:])
            else:
                imgs = imgs[...,:3]*imgs[...,-1:]
        
        self.H, self.W = H, W
        self.i_split = {
            'train': i_split[0],
            'val': i_split[1],
            'test': i_split[2]
        }
        self.focal = focal
        # self.K = np.array([
        #     [focal, 0, 0.5*W],
        #     [0, focal, 0.5*H],
        #     [0, 0, 1]
        # ])

        self.imgs = torch.FloatTensor(imgs)
        self.poses = torch.FloatTensor(poses)

    
    def get_sub_set(self, split_set: str):
        sub_set = copy.copy(self)
        sub_set.imgs = sub_set.imgs[self.i_split[split_set]]
        sub_set.poses = sub_set.poses[self.i_split[split_set]]
        return sub_set

    def get_render_set(self, phi: float=30.0, radius: float=4.0, n_frame: int=40):
        render_poses = torch.stack([pose_spherical(angle, -phi, radius)
                                for angle in np.linspace(-180,180,n_frame+1)[:-1]], 0)
        render_set = copy.copy(self)
        render_set.imgs = None
        render_set.poses = render_poses
        return render_set