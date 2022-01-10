import torch
from torch import Tensor

@torch.jit.script
def bbox2voxels(bbox: Tensor, voxel_size: float):
    '''
    bbox2voxel: https://github.com/facebookresearch/NSVF/fairnr/modules/encoder.py#L1053
    bbox: array [min_x,y,z, max_x,y,z]
    '''
    v_min, v_max = bbox[:3], bbox[3:]
    steps = ((v_max - v_min) / voxel_size).round().long() + 1
    # note the difference between torch.meshgrid and np.meshgrid. torch.meshgrid is better.
    points = torch.stack(torch.meshgrid([torch.arange(s) for s in steps]), -1) # s_x,s_y,s_z,3
    points = points * voxel_size + v_min
    return points