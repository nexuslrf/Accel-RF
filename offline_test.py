from accelRF.raysampler import NeRFRaySampler
from accelRF.pointsampler import NeRFPointSampler
from accelRF.models import PositionalEncoding, NeRF
from accelRF.render.nerf_render import NeRFRender
from accelRF.datasets import Blender
from torch.utils.data import DataLoader
import torch
import numpy as np


from accelRF.raysampler.utils import get_rays_uv, get_rays

to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

@torch.no_grad()
def render_image(H, W, K, c2w, renderer, chunk=None, device="cpu"):
    uv = torch.stack(torch.meshgrid(
                        torch.linspace(0, H-1, H, device=device), 
                        torch.linspace(0, W-1, W, device=device)),
                    -1).flip(-1).reshape(-1, 2)
    K = K.to(device)
    c2w = c2w.to(device)
    rays_o, rays_d = get_rays_uv(uv, K, c2w)
    if chunk is None:
        rgb = renderer(rays_o, rays_d)['rgb'].cpu().numpy()
    else:
        print(chunk)
        rgb = []
        for i in range(0, rays_o.shape[0], chunk):
            rgb.append(renderer(rays_o[i:i+chunk], rays_d[i:i+chunk])['rgb'].cpu().numpy())
        rgb = np.concatenate(rgb)
    rgb = to8b(rgb).reshape(H, W, 3)
    return rgb


datapath, scene = '/home/lihaoda/Accel-RF/blender_dataset/', 'lego'
multires, multires_views = 10, 4
N_samples, N_importance = 64, 128
input_ch, input_ch_views = (2*multires+1)*3, (2*multires_views+1)*3
# prepare data
dataset = Blender(datapath, scene)
test_raysampler = NeRFRaySampler(dataset.get_sub_set('test'), full_rendering=True)
test_rayloader = DataLoader(test_raysampler, num_workers=1, pin_memory=True)
# create model
nerf_render = NeRFRender(
    embedder_pts=PositionalEncoding(multires),
    embedder_views=PositionalEncoding(N_freqs=multires_views),
    point_sampler=NeRFPointSampler(
        N_samples=N_samples, N_importance=N_importance, 
        near=dataset.near, far=dataset.far, perturb=1, lindisp=True),
    model=NeRF(in_ch_pts=input_ch, in_ch_dir=input_ch_views), # coarse model
    fine_model=NeRF(in_ch_pts=input_ch, in_ch_dir=input_ch_views),
    white_bkgd=True).cuda()
nerf_render = torch.nn.DataParallel(nerf_render.jit_script())

ckpt = torch.load("../Accel-RF/200000.pt")
key_to_remove = [x for x in ckpt['state_dict'].keys() if x.startswith("module.point_sampler") or x.startswith("module.use_viewdirs")]
for k in key_to_remove:
    del ckpt['state_dict'][k]
nerf_render.load_state_dict(ckpt['state_dict'])
nerf_render.eval()


H, W, f = dataset.get_hwf()
K = dataset.get_K(0)[None,:]
c2w = dataset[0]['extrinsics'][None,:]

from PIL import Image
rgb = render_image(400, 600, K, c2w, nerf_render, chunk=4800, device=0)
Image.fromarray(rgb).save("test.png")
