import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import imageio
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from accelRF.datasets import Blender
from accelRF.metrics.nsvf_loss import nsvf_loss
from accelRF.models import (NSVF_MLP, BackgroundField, PositionalEncoding,
                            VoxelEncoding)
from accelRF.pointsampler import NSVFPointSampler
from accelRF.raysampler import (PerViewRaySampler, RenderingRaySampler,
                                VoxIntersectRaySampler)
from accelRF.render.nsvf_render import NSVFRender
from accelRF.rep.voxel_grid import VoxelGrid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from opts import config_parser

parser = config_parser()
args = parser.parse_args()
n_gpus = torch.cuda.device_count()
n_replica = 1
device = 'cuda'
cudnn.benchmark = True
savedir = os.path.join(args.basedir, args.expname)

if args.local_rank >= 0:
    dist.init_process_group(backend='nccl', init_method="env://")
    device = f'cuda:{args.local_rank}'
    n_replica = n_gpus

if args.local_rank <= 0:
    tb_writer = SummaryWriter(os.path.join(args.basedir, 'summaries', args.expname))

# parameters
if args.pruning_every_steps>0:
    args.pruning_steps = list(range(0, args.N_iters, args.pruning_every_steps))[1:]
args.loss_weights = {
    'rgb': args.loss_w_rgb, 'alpha': args.loss_w_alpha, 'reg_term': args.loss_w_reg
    }

img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = torch.no_grad()(lambda x : -10. * torch.log10(x))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

def main():
    # prepare data
    if args.dataset_type == 'blender':
        dataset = dataset = Blender(args.datadir, args.scene, args.half_res, 
                                args.testskip, args.white_bkgd, with_bbox=True)
    else:
        print(f'{args.dataset_type} has not been supported yet...')
        exit()
    vox_grid = VoxelGrid(dataset.bbox, args.voxel_size).to(device)

    # create model
    input_ch, input_ch_views = (2*args.multires+1)*args.embed_dim, (2*args.multires_views)*3 # view not include x
    nsvf_render = NSVFRender(
        point_sampler=NSVFPointSampler(args.step_size),
        pts_embedder=PositionalEncoding(args.multires, pi_bands=True),
        view_embedder=PositionalEncoding(args.multires_views, angular_enc=True, include_input=False),
        voxel_embedder=VoxelEncoding(vox_grid.n_corners, args.embed_dim),
        model=NSVF_MLP(in_ch_pts=input_ch, in_ch_dir=input_ch_views, D_rgb=args.D_rgb, W_rgb=args.W_rgb, 
            D_feat=args.D_feat, W_feat=args.W_feat, D_sigma=args.D_sigma, W_sigma=args.W_sigma, 
            layernorm=(not args.no_layernorm), with_reg_term=(args.loss_w_reg!=0)),
        vox_rep=vox_grid,
        bg_color=BackgroundField(bg_color=1. if args.white_bkgd else 0., trainable=False) if args.bg_field else None,
        white_bkgd=args.white_bkgd,
        min_color=args.min_color,
        early_stop_thres=args.early_stop_thres,
        chunk=args.chunk * n_gpus if args.local_rank < 0 else args.chunk,
    ).to(device)
        
    # create optimizer
    optimizer  = optim.Adam(nsvf_render.parameters(), args.lrate, betas=(0.9, 0.999))
    start = 0
    # load checkpoint
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(savedir, f) for f in sorted(os.listdir(savedir)) if f.endswith('.pt')]
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Load from: ', ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=device)
        start = ckpt['global_step']
        n_voxels, n_corners, grid_shape = ckpt['n_voxels'], ckpt['n_corners'], ckpt['grid_shape']
        if n_voxels != vox_grid.n_voxels:
            vox_grid.load_adjustment(n_voxels, grid_shape)
            nsvf_render.voxel_embedder.load_adjustment(n_corners)
            optimizer.param_groups.clear(); optimizer.state.clear()
            optimizer.add_param_group({'params':nsvf_render.parameters()})
        try:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            optimizer.param_groups[0]['initial_lr'] = args.lrate
        except:
            pass
        nsvf_render.load_state_dict(ckpt['state_dict'])

    lr_sched = optim.lr_scheduler.ExponentialLR(optimizer, 0.1**(1/(args.lrate_decay*1000)), last_epoch=start-1)
    
    # parallelization
    if args.local_rank >=0:
        nsvf_render_ = torch.nn.parallel.DistributedDataParallel(nsvf_render, device_ids=[args.local_rank])
    else:
        nsvf_render_ = nn.DataParallel(nsvf_render)
    # nsvf_render_ = nsvf_render

    # prepare dataloader
    train_base_raysampler = \
        PerViewRaySampler(dataset.get_sub_set('train'), args.N_rand, args.N_iters, args.N_views, 
            precrop=False, full_rays=args.full_rays, start_epoch=start, rank=args.local_rank, n_replica=n_replica)
    train_vox_raysampler = VoxIntersectRaySampler(args.N_rand, train_base_raysampler, vox_grid, device=device)
    test_raysampler = VoxIntersectRaySampler(0, RenderingRaySampler(dataset.get_sub_set('test')), 
                            vox_grid, mask_sample=False, device=device, num_workers=1)
    val_raysampler = VoxIntersectRaySampler(0, RenderingRaySampler(dataset.get_sub_set('val')), 
                            vox_grid, mask_sample=False, device=device, num_workers=0)

    train_rayloader = DataLoader(train_vox_raysampler, num_workers=0) # vox_raysampler's N_workers==0, pin_mem==False
    test_rayloader = DataLoader(test_raysampler, num_workers=0)
    train_ray_iter = iter(train_rayloader)

    # start training
    for i in trange(start, args.N_iters):
        # get one training batch
        ray_batch = next(train_ray_iter)
        rays_o, rays_d = ray_batch['rays_o'][0], ray_batch['rays_d'][0]
        vox_idx, t_near, t_far = ray_batch['vox_idx'][0], ray_batch['t_near'][0], ray_batch['t_far'][0]
        hits, gt_rgb = ray_batch['hits'][0], ray_batch['gt_rgb'][0]

        render_out = nsvf_render_(rays_o, rays_d, vox_idx, t_near, t_far, hits)
        loss, sub_losses = nsvf_loss(render_out, gt_rgb, args.loss_weights)
        optimizer.zero_grad()
        if loss.grad_fn is not None:
            loss.backward()
            optimizer.step()

        lr_sched.step()

        if i%args.i_print==0:
            psnr = mse2psnr(sub_losses['rgb'])
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()} " + \
                    # f"Alpha: {sub_losses['alpha'].item()} " + \
                    f"Hit ratio: {hits.sum().item()/hits.shape[0]} " + \
                    f"Bg ratio: {(gt_rgb.sum(-1).eq(0)).sum().item()/hits.shape[0]}")
            if args.local_rank <= 0:
                tb_writer.add_scalar('loss', loss, i)
                tb_writer.add_scalar('psnr', psnr, i)
                if i%args.i_img==0:
                    # Log a rendered validation view to Tensorboard
                    img_i=torch.randint(len(val_raysampler), ())
                    ray_batch = val_raysampler[img_i]
                    rays_o, rays_d = ray_batch['rays_o'], ray_batch['rays_d']
                    vox_idx, t_near, t_far = ray_batch['vox_idx'], ray_batch['t_near'], ray_batch['t_far']
                    hits, gt_rgb = ray_batch['hits'], ray_batch['gt_rgb']
                    with torch.no_grad():
                        nsvf_render_.eval()
                        render_out = nsvf_render_(rays_o, rays_d, vox_idx, t_near, t_far, hits)
                        nsvf_render_.train()
                        psnr = mse2psnr(img2mse(render_out['rgb'], gt_rgb))
                    tb_writer.add_scalar('psnr_eval', psnr, i)
                    H, W, _ = dataset.get_hwf()
                    tb_writer.add_image('gt_rgb', gt_rgb.reshape(H,W,-1), i, dataformats="HWC")
                    tb_writer.add_image('rgb', to8b(render_out['rgb'].cpu().numpy()).reshape(H,W,-1), i, dataformats='HWC')
                    tb_writer.add_image('disp', render_out['disp'].reshape(H,W), i, dataformats="HW")
                    tb_writer.add_image('acc', render_out['acc'].reshape(H,W), i, dataformats="HW")
                                      
        if i%args.i_testset == 0 and i > 0 and args.local_rank <= 0:
            eval(nsvf_render_, test_rayloader, dataset.get_hwf()[:2], os.path.join(savedir, f'testset_{i:06d}'))
        
        if (i+1)%args.i_weights==0 and args.local_rank <= 0:
            path = os.path.join(savedir, f'{i+1:06d}.pt')
            torch.save({
                'global_step': i+1,
                'state_dict': nsvf_render.state_dict(), # keys start with 'module'
                'optimizer_state_dict': optimizer.state_dict(),
                'n_voxels': vox_grid.n_voxels.item(), 'n_corners': vox_grid.n_corners.item(),
                'grid_shape': vox_grid.grid_shape.tolist()
            }, path)

        # model refinement
        reset_module = False
        if i in args.pruning_steps:
            done = nsvf_render.pruning(thres=args.pruning_thres)
            reset_module = done or reset_module
        if i in args.splitting_steps:
            done = nsvf_render.splitting()
            reset_module = done or reset_module
        if i in args.half_stepsize_steps:
            nsvf_render.half_stepsize()
        if reset_module:
            if args.local_rank >= 0:
                del nsvf_render_
                nsvf_render_ = nn.parallel.DistributedDataParallel(nsvf_render, device_ids=[args.local_rank])
            optimizer.zero_grad()
            # https://discuss.pytorch.org/t/delete-parameter-group-from-optimizer/46814/8
            optimizer.param_groups.clear() # optimizer.param_group = []
            optimizer.state.clear() # optimizer.state = defaultdict(dict)
            optimizer.add_param_group({'params':nsvf_render.parameters()}) # necessary!


def eval(nsvf_render, rayloader, img_hw, testsavedir):
    os.makedirs(testsavedir, exist_ok=True) 
    nsvf_render.eval()
    for i, ray_batch in enumerate(tqdm(rayloader)):
        rays_o, rays_d = ray_batch['rays_o'][0], ray_batch['rays_d'][0]
        vox_idx, t_near, t_far = ray_batch['vox_idx'][0], ray_batch['t_near'][0], ray_batch['t_far'][0]
        hits, gt_rgb = ray_batch['hits'][0], ray_batch['gt_rgb'][0]
        with torch.no_grad():
            render_out = nsvf_render(rays_o, rays_d, vox_idx, t_near, t_far, hits)
        psnr = mse2psnr(img2mse(gt_rgb, render_out['rgb']))
        imageio.imwrite(os.path.join(testsavedir, f'{i:03d}.png'), 
                    to8b(render_out['rgb'].cpu().numpy()).reshape(*img_hw,-1))
        tqdm.write(f"[Test] #: {i} PSNR: {psnr.item()}")
    nsvf_render.train()

if __name__ == '__main__':
    print("Args: \n", args, "\n", "-"*40)
    os.makedirs(savedir, exist_ok=True)
    f = os.path.join(savedir, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(savedir, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())
    # start main program
    main()
