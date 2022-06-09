import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from accelRF.datasets import Blender
from accelRF.raysampler import PerViewRaySampler, VoxIntersectRaySampler
from accelRF.pointsampler import NSVFPointSampler
from accelRF.models import PositionalEncoding, VoxelEncoding, NSVF_MLP, BackgroundField
from accelRF.render.nsvf_render import NSVFRender
from accelRF.rep.voxel_grid import VoxelGrid
from accelRF.metrics.nsvf_loss import nsvf_loss
# parameters
datapath, scene = '/data/stu01/ruofan/nerf-experiment/data/nerf_synthetic/', 'lego'
N_rand, N_iters, N_views = 2048, 100000, 4
multires, multires_views = 6, 4
N_samples = 64, 128
input_ch, input_ch_views = (2*multires+1)*32, (2*multires_views)*3
voxel_size = 0.4
step_size = 0.125
embed_dim = 32
loss_weights = {'rgb': 10., 'alpha': 0.1, 'reg_term': 0}
pruning_thres = 0.01
pruning_steps = []
splitting_steps = [4]
half_stepsize_steps = []
local_rank = -1
# prepare data
dataset = Blender(datapath, scene, with_bbox=True, half_res=True)
vox_grid = VoxelGrid(dataset.bbox, voxel_size).cuda()
# ray sampler
base_raysampler = PerViewRaySampler(dataset.get_sub_set('train'), N_rand*4, N_iters, N_views, precrop=False)
vox_raysampler = VoxIntersectRaySampler(N_rand, base_raysampler, vox_grid)
train_rayloader = DataLoader(vox_raysampler, num_workers=0) # vox_raysampler's N_workers==0, pin_mem==False
# create model
nsvf_render = NSVFRender(
    point_sampler=NSVFPointSampler(step_size),
    pts_embedder=PositionalEncoding(multires, pi_bands=True),
    view_embedder=PositionalEncoding(multires_views, angular_enc=True, include_input=False),
    voxel_embedder=VoxelEncoding(vox_grid.n_corners, embed_dim),
    model=NSVF_MLP(in_ch_pts=input_ch, in_ch_dir=input_ch_views),
    vox_rep=vox_grid,
    bg_color=None
).cuda()
nsvf_render_ = nn.DataParallel(nsvf_render)
# nsvf_render_ = nsvf_render
# create optimizer
optimizer  = torch.optim.Adam(nsvf_render.parameters(), 1e-3)
# start training
for i, ray_batch in enumerate(train_rayloader):
    # TODO
    print('Run: ', nsvf_render.voxel_embedder.get_weight().sum())
    rays_o, rays_d = ray_batch['rays_o'][0], ray_batch['rays_d'][0]
    vox_idx, t_near, t_far = ray_batch['vox_idx'][0], ray_batch['t_near'][0], ray_batch['t_far'][0]
    hits, gt_rgb = ray_batch['hits'][0], ray_batch['gt_rgb'][0]
    render_out = nsvf_render_(rays_o, rays_d, vox_idx, t_near, t_far, hits)
    loss, _ = nsvf_loss(render_out, gt_rgb, loss_weights)
    optimizer.zero_grad()
    if loss.grad_fn is not None:
        loss.backward()
        print("Grad norm: ", nsvf_render.model.rgb_layers[0].net[4].weight.grad.norm())
        optimizer.step()
    # print('loss: ', loss)
    reset_module = False
    if i in pruning_steps:
        done = nsvf_render.pruning(thres=pruning_thres)
        reset_module = done or reset_module
    if i in splitting_steps:
        done = nsvf_render.splitting()
        reset_module = done or reset_module
    if i in half_stepsize_steps:
        nsvf_render.half_stepsize()
    if reset_module:
        if local_rank >= 0:
            del nsvf_render_
            nsvf_render_ = nn.parallel.DistributedDataParallel(...)
        optimizer.zero_grad()
        optimizer.param_groups.clear() # optimizer.param_group = []
        optimizer.state.clear() # optimizer.state = defaultdict(dict)
        optimizer.add_param_group({'params':nsvf_render_.parameters()}) # necessary!

    if i>=12:
        break