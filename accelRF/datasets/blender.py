import os
import json
import numpy as np
import imageio
import cv2
import tqdm
import torch
import copy
from .base import BaseDataset


__all__ = ['Blender']

# TODO maybe you can add auto data downloading?

# this bounding box values are copy from KiloNeRF
# https://github.com/creiser/kilonerf/tree/master/data/nsvf/Synthetic_NeRF/
# Note: unless specified, these bounding box values are not used in normal NeRF model.
# format: [min_x,y,z, max_x,y,z]
bounding_box = {
    'lego':   torch.tensor([-0.67, -1.2, -0.37, 0.67, 1.2, 1.03]),
    'hotdog': torch.tensor([-1.28349, -1.34831, -0.376072, 1.28349, 1.21868, 0.473294]),
    'ficus': torch.tensor([-0.501468, -0.894687, -1.08433, 0.622253, 0.653115, 1.22525]),
    'drum':  torch.tensor([-1.20256, -0.849854, -0.595211, 1.20256, 1.05901, 1.08823]),
    'chair': torch.tensor([-0.85, -0.8, -1.1, 0.85, 0.85, 1.1]),
    'material': torch.tensor([-1.19772, -0.882472, -0.311908, 1.14904, 1.05773, 0.311908]),
    'mic':  torch.tensor([-1.39144, -0.963949, -0.822599, 0.963776, 1.26331, 1.29303]),
    'ship': torch.tensor([-1.38519, -1.37695, -0.636088, 1.46763, 1.47587, 0.79542]),
}
#######

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
        half_res: bool = False, testskip: int = 1, white_bkgd: bool = False,
        with_bbox: bool = False
    )->None:
        super().__init__()
        self.scene = scene
        self.with_bbox = with_bbox
        basedir = os.path.join(root, scene)
        splits = ['train', 'test', 'val']
        metas = {}
        for s in splits:
            json_file = os.path.join(basedir, 'transforms_{}.json'.format(s))
            if os.path.exists(json_file):
                with open(json_file, 'r') as fp:
                    metas[s] = json.load(fp)
        splits = metas.keys()

        self.img_paths = []
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

            for frame in tqdm.tqdm(meta['frames'][::skip]):
                fname = os.path.join(basedir, frame['file_path'] + '.png')
                self.img_paths.append(fname)
                imgs.append(imageio.imread(fname))
                poses.append(np.array(frame['transform_matrix']))
            imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
            poses = np.array(poses).astype(np.float32)
            counts.append(counts[-1] + imgs.shape[0])
            all_imgs.append(imgs)
            all_poses.append(poses)

        i_split = [np.arange(counts[i], counts[i+1]) for i in range(len(splits))]

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
        
        self.H, self.W = H, W
        self.i_split = {
            'train': i_split[0],
            'test': i_split[1],
            'val': i_split[2] if 'val' in splits else i_split[1]
        }
        self.focal = focal

        if imgs.shape[-1] == 4:
            # if keep_alpha:
            #     self.alpha = torch.FloatTensor(imgs[...,-1:]) # [N, H, W, 1]
            if white_bkgd:
                imgs = imgs[...,:3]*imgs[...,-1:] + (1.-imgs[...,-1:])
            else:
                imgs = imgs[...,:3]*imgs[...,-1:]

        self.imgs = torch.FloatTensor(imgs) # [N, H, W, 3]
        self.Ts = torch.FloatTensor(poses) # [N, 3, 4]
        # represent as an intrinsic matrix, might be useful later.
        self.Ks = torch.FloatTensor([[0.5*W, 0.5*H, focal, focal]]) # [1, 4]

        if with_bbox:
            self.bbox = bounding_box[scene]

    def get_render_set(self, n_frame: int=40, phi: float=30.0, radius: float=4.0):
        render_poses = torch.stack([pose_spherical(angle, -phi, radius)
                                for angle in np.linspace(-180,180,n_frame+1)[:-1]], 0)
        render_set = copy.copy(self)
        render_set.imgs = None
        render_set.Ts = render_poses
        return render_set