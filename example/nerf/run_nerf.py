# this is minimal implementation of orginal NeRF, for the Test purpose.
# TODO more options will be gradually added. 🙂
# To run this example: python example/nerf/run_nerf.py --config example/configs/lego.txt

# there are two ways to render final output:
# 1. batchify NN inference, then concate them and feed them to render
# 2. process NN + render batch by batch 
# Which is better? 🤔
# I choose the second option to better utilize multi-GPU)
# TODO Write program both ways and test speed.
import os, sys
import logging

logging.basicConfig(level=logging.INFO)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from opts import config_parser
from tqdm import tqdm, trange

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from accelRF.datasets import Blender
from accelRF.raysampler import NeRFRaySampler
from accelRF.pointsampler import NeRFPointSampler
from accelRF.models import PositionalEncoding, NeRF
from accelRF.render.nerf_render import NeRFRender

parser = config_parser()
args = parser.parse_args()
device = 'cuda'

def main():
    # prepare data
    dataset = Blender(args.datadir, args.scene, args.half_res, args.testskip, args.white_bkgd)

    train_raysampler = NeRFRaySampler(dataset.get_sub_set('train'), args.N_rand, args.N_iters,
            use_batching=(not args.no_batching), use_ndc=(not args.no_ndc), precrop=(args.precrop_iters > 0), 
            precrop_frac=args.precrop_frac, precrop_iters=args.precrop_iters)
    test_raysampler = NeRFRaySampler(dataset.get_sub_set('test'), full_rendering=True)
    train_rayloader = DataLoader(train_raysampler, num_workers=1, pin_memory=True)
    test_rayloader = DataLoader(test_raysampler, num_workers=1, pin_memory=True)
    # create model
    input_ch, input_ch_views = (2*args.multires+1)*3, (2*args.multires_views+1)*3
    nerf_render = NeRFRender(
        embedder_pts=PositionalEncoding(N_freqs=args.multires) if args.i_embed==0 else None,
        embedder_views=PositionalEncoding(N_freqs=args.multires_views) if args.i_embed==0 else None,
        point_sampler=NeRFPointSampler(
            N_samples=args.N_samples, N_importance=args.N_importance, 
            near=dataset.near, far=dataset.far, perturb=args.perturb, lindisp=args.lindisp),
        model=NeRF(in_ch_pts=input_ch, in_ch_dir=input_ch_views,
            D=args.netdepth, W=args.netwidth, skips=[4]), # coarse model
        fine_model=NeRF(in_ch_pts=input_ch, in_ch_dir=input_ch_views, 
            D=args.netdepth, W=args.netwidth, skips=[4]) if args.N_importance > 0 else None,
        white_bkgd=args.white_bkgd
    )
    # nerf_render = nn.DataParallel(nerf_render).to(device)
    nerf_render = nerf_render.to(device)
    # create optimizer
    optimizer  = optim.Adam(nerf_render.parameters(), args.lrate, betas=(0.9, 0.999))
    lr_sched = optim.lr_scheduler.ExponentialLR(optimizer, 0.1**(1/(args.lrate_decay*1000)), last_epoch=-1)
    # def metrics TODO metrics can be pre-defined in a accelRF collection
    img2mse = lambda x, y : torch.mean((x - y) ** 2)
    mse2psnr = torch.no_grad()(lambda x : -10. * torch.log10(x))

    train_ray_iter = iter(train_rayloader)
    start = 0
    for i in trange(start, args.N_iters):
        # get one training batch
        ray_batch = next(train_ray_iter)
        rays_o, rays_d = ray_batch['rays_o'][0].to(device), ray_batch['rays_d'][0].to(device)
        gt_rgb = ray_batch['gt_rgb'][0].to(device)
        # TODO you can add an inner loop for further batchifying.
        render_out = nerf_render(rays_o, rays_d)
        img_loss = img2mse(gt_rgb, render_out['rgb'])
        psnr = mse2psnr(img_loss)
        loss = img_loss
        if 'rgb0' in render_out:
            img_loss0 = img2mse(gt_rgb, render_out['rgb0'])
            psnr0 = mse2psnr(img_loss0)
            loss = loss + img_loss0

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr_sched.step()

        if i%args.i_testset == 0 and i > 0:
            nerf_render.eval()
            for i, ray_batch in enumerate(tqdm(test_rayloader)):
                rays_o, rays_d = ray_batch['rays_o'][0].to(device), ray_batch['rays_d'][0].to(device)
                gt_rgb = ray_batch['gt_rbg'][0].to(device)
                render_out = nerf_render(rays_o, rays_d)
                img_loss = img2mse(gt_rgb, render_out['rgb'])
                psnr = mse2psnr(img_loss)
        
        if i%args.i_print==0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")

if __name__ == '__main__':
    main()
