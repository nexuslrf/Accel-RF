# Accel-RF

## Basic framework for NeRF

### Basic Features: 
* [x] Based on PyTorch.
* [x] Modularized design.
* [x] Support JIT optimization.
* [x] Support multi-GPU training.
* [ ] amp features **WIP**

### Supported Models

* [x] Vanilla NeRF
* [x] NSVF
* [x] VolSDF
* [ ] ... 

### Framework:

The whole NeRF training workflow can be decomposed into major 5 modules/steps.

<img src="https://bl3301files.storage.live.com/y4m_heZ4ahhH9_qTzlk2QJkWemhV3RqQoufrAhalXfsgqDiymrANqBPijViY8VXkFC70yld6OdaXOPYcSnpPAmeTn_v2O02PNS-MkvAiqTXe1odbhLDuJHajKfWm2wKBKnPLLiab-OXJtK8UciCpA3A18HUp3dbfMb4eGZkaGVPeOCi2cdUyR0eYSGWhaeC2-X_?width=3854&height=847&cropmode=none"/>

**Datasets**

* NeRF's dataset is a set of images (with camera poses) around a scene. Unlike datasets for other DL training tasks, this dataset does not output data that can be directly used by NN models. Given an index i, the dataset just outputs i-th image and its corresponding camera pose. The outputs are then be processed by RaySampler and PointSampler sequentially.
* Currently supported dataset:
    * [x] NeRF Blender
    * [x] LLFF
    * [x] BlendedMVS (in volsdf format) 
    * [ ] To be added…

**RaySampler**
* RaySampler is a wrapper around the dataset. It samples camera rays from the images inside the dataset. The rays are represented in the form of  r = o + dt, thus rays_o and rays_d are the two outputs of the RaySampler.
* RaySampler is further wrapper with PyTorch's dataloader for multi-worker sampling during training. 
* Currently supported sampling methods:
    * [x] Randomly sampled rays on one randomly picked image.
    * [x] Randomly sampled rays from rays of all images (use_batching=True).

**PointSampler**
* Based on rays_o and rays_d from RaySampler, PointSampler samples xyz points along the ray, and these sampled points are finally feeded into NN models for radiance field prediction.
* Importance-based sampling also relies on the predicted weights from the coarse NeRF model.
* Currently supported sampling methods:
    * [x] Uniformly sampling along the ray --> coarse sampling.
    * [x] Importance sampling --> hierarchical sampling (fine)

**NN Models**
* Neural network components, including:
    * [x] NeRF's MLP
    * [x] PositionalEncoding module
* Note that the outputs of NN still require further processing (ray-tracing integration).

**Renderer**
* Performs ray-based integration to convert NN's raw output into RGB color (volume rendering).
* Provides a wrapper to combine PointSampler, Encoding module, NN models into one meta nn.Module. 
    * [x] This can ease the parameter management and multi-GPU parallelism.

Example

A 40-line simplified NeRF training (the complete example is [here](example/nerf/))
```python
import torch
from torch.utils.data import DataLoader
from accelRF.datasets import Blender
from accelRF.raysampler import NeRFRaySampler
from accelRF.pointsampler import NeRFPointSampler
from accelRF.models import PositionalEncoding, NeRF
from accelRF.render.nerf_render import NeRFRender
# parameters
datapath, scene = 'path/to/blender', 'lego'
N_rand, N_iters = 1024, 100000
multires, multires_views = 10, 4
N_samples, N_importance = 64, 128
input_ch, input_ch_views = (2*multires+1)*3, (2*multires_views+1)*3
# prepare data
dataset = Blender(datapath, scene)
# create model
nerf_render = NeRFRender(
    embedder_pts=PositionalEncoding(N_freqs=multires),
    embedder_views=PositionalEncoding(N_freqs=multires_views),
    point_sampler=NeRFPointSampler(
        N_samples=N_samples, N_importance=N_importance, near=dataset.near, far=dataset.far),
    model=NeRF(in_ch_pts=input_ch, in_ch_dir=input_ch_views), # coarse model
    fine_model=NeRF(in_ch_pts=input_ch, in_ch_dir=input_ch_views) if N_importance > 0 else None
)
# create optimizer
optimizer  = torch.optim.Adam(nerf_render.parameters(), lr=5e-4)
# create ray sampler
train_raysampler = NeRFRaySampler(dataset.get_sub_set('train'), N_rand, N_iters)
train_rayloader = DataLoader(train_raysampler, num_workers=1, pin_memory=True)
# start training
for i, ray_batch in enumerate(train_rayloader):
    rays_o, rays_d = ray_batch['rays_o'][0].cuda(), ray_batch['rays_d'][0].cuda()
    gt_rgb = ray_batch['gt_rgb'][0].cuda()
    render_out = nerf_render(rays_o, rays_d)
    loss = ((gt_rgb - render_out['rgb'])**2).mean()
    if 'rgb0' in render_out:
        loss = loss + ((gt_rgb - render_out['rgb0'])**2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```
