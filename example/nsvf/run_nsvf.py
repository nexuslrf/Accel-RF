import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from torch.utils.data import DataLoader
from accelRF.datasets import Blender
from accelRF.raysampler import PerViewRaySampler, VoxIntersectRaySampler
from accelRF.pointsampler import NSVFPointSampler
from accelRF.models import PositionalEncoding, VoxelEncoding, NSVF_MLP
from accelRF.render.nsvf_render import NSVFRender
from accelRF.rep.voxel_grid import VoxelGrid
# parameters
datapath, scene = '/data/stu01/ruofan/nerf-experiment/data/nerf_synthetic/', 'lego'
N_rand, N_iters, N_views = 1024, 100000, 4
multires, multires_views = 10, 4
N_samples, N_importance = 64, 128
input_ch, input_ch_views = (2*multires+1)*32, (2*multires_views+1)*3
voxel_size = 0.4
step_size = 0.125
embed_dim = 32
# prepare data
dataset = Blender(datapath, scene, with_bbox=True)
vox_grid = VoxelGrid(dataset.bbox, voxel_size).cuda()
# ray sampler
base_raysampler = PerViewRaySampler(dataset.get_sub_set('train'), N_rand*4, N_iters, N_views, precrop=False)
vox_raysampler = VoxIntersectRaySampler(N_rand, base_raysampler, vox_grid)
train_rayloader = DataLoader(vox_raysampler, num_workers=1, pin_memory=True)
# create model
nsvf_render = NSVFRender( # TODO
    point_sampler=NSVFPointSampler(step_size),
    embedder_pts=PositionalEncoding(N_freqs=multires),
    embedder_views=PositionalEncoding(N_freqs=multires_views, include_input=False),
    voxel_embedding=VoxelEncoding(vox_grid, embed_dim),
    model=NSVF_MLP(in_ch_pts=input_ch, in_ch_dir=input_ch_views), # identify what changes we need to make.
).cuda()
# create optimizer
optimizer  = torch.optim.Adam(nsvf_render.parameters(), lr=5e-4)
# start training
for i, ray_batch in enumerate(train_rayloader):
    # TODO
    rays_o, rays_d = ray_batch['rays_o'][0].cuda(), ray_batch['rays_d'][0].cuda()
    gt_rgb = ray_batch['gt_rgb'][0].cuda()
    render_out = nsvf_render(rays_o, rays_d)
    loss = ((gt_rgb - render_out['rgb'])**2).mean() # TODO, nsvf has more complicated loss
    if 'rgb0' in render_out:
        loss = loss + ((gt_rgb - render_out['rgb0'])**2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()