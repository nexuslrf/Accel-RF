import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

class OrbitCamera:
    def __init__(self, H, W, r=2, fovy=30):
        self.W = W
        self.H = H
        self.radius = r # camera distance from center
        self.fovy = fovy
        self.center = np.array([0, 0, 0], dtype=np.float32)
        self.rot = R.from_quat([0, 0, 0, 1]) # scalar last
        self.up = np.array([0, 1, 0], dtype=np.float32)

    @property
    def pose(self):
        # first move camera to radius
        res = np.eye(4, dtype=np.float32)
        res[2, 3] -= self.radius
        # rotate
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res
        # translate
        res[:3, 3] -= self.center
        return res

    def get_pose_text(self):
        pose = self.pose
        return np.array2string(pose, formatter={'float_kind':lambda x: "%.2f" % x})




    def get_rays(self, device="cpu"):
        W, H = self.W, self.H
        focal = H / (2 * np.tan(self.fovy / 2))
        c2w = torch.tensor(self.pose)
        c2w = c2w.to(device)
        i, j = torch.meshgrid(
            torch.linspace(0, W-1, W, device=device),
            torch.linspace(0, H-1, H, device=device))  # pytorch's meshgrid has indexing='ij'
        i = i.t()
        j = j.t()
        dirs = torch.stack([(i-W*.5)/focal, -(j-H*.5)/focal, torch.ones_like(i)], -1)
        # Rotate ray directions from camera frame to the world frame
        rays_d = torch.sum(dirs[..., None, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
        # Translate camera frame's origin to the world frame. It is the origin of all rays.
        rays_o = c2w[:3,-1].expand(rays_d.shape)
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        return rays_o, rays_d

    def rotate(self, dx, dy):
        side = self.rot.as_matrix()[:3, 0]
        rotvec_x = self.up * np.radians(-0.1 * dx)
        rotvec_y = side * np.radians(0.1 * dy)
        self.rot = R.from_rotvec(rotvec_x) * R.from_rotvec(rotvec_y) * self.rot


    def scale(self, delta):
        self.radius *= 1.1 ** (-delta)

    def pan(self, dx, dy, dz=0):
        self.center += 0.001 * self.rot.as_matrix()[:3, :3] @ np.array([-dx, -dy, dz])
