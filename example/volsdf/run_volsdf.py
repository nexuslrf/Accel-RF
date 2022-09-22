import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from opts import config_parser
from tqdm import tqdm, trange
import imageio
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from accelRF.datasets import Blender, SceneDataset, NSVFDataset
from accelRF.raysampler import RenderingRaySampler, PerViewRaySampler
from accelRF.pointsampler.volsdf_pointsampler import VolSDFPointSampler
from accelRF.models import PositionalEncoding
from accelRF.models.volsdf import AbsDensity, SDFNet, RGBNet, LaplaceDensity
from accelRF.render.volsdf_render import VolSDFRender

parser = config_parser()
args = parser.parse_args()
n_gpus = torch.cuda.device_count()
n_replica = 1
device = 'cuda'
cudnn.benchmark = True
savedir = os.path.join(args.basedir, args.expname)

if args.local_rank >= 0:
    torch.cuda.set_device(args.local_rank)
    device = f'cuda:{args.local_rank}'
    dist.init_process_group(backend='nccl', init_method="env://")
    n_replica = n_gpus

if args.local_rank <= 0:
    tb_writer = SummaryWriter(os.path.join(args.basedir, 'summaries', args.expname))

l1loss = lambda x,y: torch.abs(x-y).mean()
eikonal_loss = lambda g: ((g.norm(2, dim=1) - 1) ** 2).mean()
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = torch.no_grad()(lambda x : -10. * torch.log10(x))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

def main():
    # prepare data
    if args.dataset_type == 'blender':
        dataset = Blender(args.datadir, args.scene, args.half_res, args.testskip)
        # args.near = dataset.near
    if args.dataset_type == 'mvs_scenes':
        dataset = SceneDataset(args.datadir, args.scene, args.testskip)
    if args.dataset_type == 'nsvf_datasets':
        dataset = NSVFDataset(args.datadir, args.scene, args.testskip)

    sdf_d_in = (2*args.multires+1)*3
    rgb_d_in = 9+2*args.multires_views*3 if args.rgb_mode == 'idr' else (2*args.multires_views+1)*3
    bg_sdf_d_in = (2*args.multires_bg+1)*4
    bg_rgb_d_in = (2*args.multires_views+1)*3

    volsdf_render = VolSDFRender(
        # VolSDF model
        point_sampler=VolSDFPointSampler(args.scene_bounding_sphere, args.near, args.far, args.N_samples, args.N_samples_eval,
            args.N_samples_extra, args.eps, args.beta_iters, args.max_total_iters, args.inverse_sphere_bg,
            args.N_samples_inverse_sphere, args.add_tiny, args.with_eikonal_samples),
        embedder_pts=PositionalEncoding(N_freqs=args.multires) if args.multires>0 else None,
        embedder_views=PositionalEncoding(N_freqs=args.multires_views) if args.multires_views>0 else None,
        sdf_net=SDFNet(args.feature_vector_size, 0.0, sdf_d_in, 1, [args.W_sdf]*args.D_sdf, 
                    args.sdf_geo_init, args.sdf_bias, args.sdf_skip_in, args.sdf_weight_norm),
        rgb_net=RGBNet(args.feature_vector_size, args.rgb_mode, rgb_d_in, 3, 
                    [args.W_rgb]*args.D_rgb, args.rgb_weight_norm),
        density_fn=LaplaceDensity(args.beta, args.beta_min),
        # NeRF++ background
        bg_embedder_pts=PositionalEncoding(N_freqs=args.multires_bg) if args.multires_bg>0 else None,
        bg_sdf_net=SDFNet(args.bg_feature_vector_size, 0.0, bg_sdf_d_in, 1, 
                        [args.W_bg_sdf]*args.D_bg_sdf, skip_in=args.bg_sdf_skip_in) if args.inverse_sphere_bg else None,
        bg_rgb_net=RGBNet(args.bg_feature_vector_size, 'nerf', bg_rgb_d_in, 3, 
                        [args.W_bg_rgb]*args.D_bg_rgb) if args.inverse_sphere_bg else None,
        bg_density_fn=AbsDensity() if args.inverse_sphere_bg else None,
        # other options
        scene_bounding_sphere=args.scene_bounding_sphere,
        with_eikonal=args.with_eikonal_samples,
        chunk=args.chunk
    ).to(device)

    # create optimizer
    optimizer  = optim.Adam(volsdf_render.parameters(), args.lrate)
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
        try:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            optimizer.param_groups[0]['initial_lr'] = args.lrate
        except:
            pass
        volsdf_render.load_state_dict(ckpt['state_dict'])

    lr_sched = optim.lr_scheduler.ExponentialLR(optimizer, 0.1**(1/(args.lrate_decay*1000)), last_epoch=start-1)
    if args.local_rank >=0:
        volsdf_render_ = torch.nn.parallel.DistributedDataParallel(volsdf_render, device_ids=[args.local_rank])
    else:
        volsdf_render_ = nn.DataParallel(volsdf_render)

    # prepare dataloader
    train_raysampler = PerViewRaySampler(
        dataset.get_sub_set('train'), args.N_rand, args.N_iters, precrop=(args.precrop_iters > 0), 
        precrop_frac=args.precrop_frac, precrop_iters=args.precrop_iters, normalize_dir=True,
        start_epoch=start, rank=args.local_rank, n_replica=n_replica) # ~~use_ndc=(not args.no_ndc)~~
    test_raysampler = RenderingRaySampler(dataset.get_sub_set('test'), normalize_dir=True)
    val_raysampler = RenderingRaySampler(dataset.get_sub_set('val'), normalize_dir=True)
    train_rayloader = DataLoader(train_raysampler, num_workers=1, pin_memory=True)
    test_rayloader = DataLoader(test_raysampler, num_workers=1, pin_memory=True)

    train_ray_iter = iter(train_rayloader)
    for i in trange(start, args.N_iters):
        # get one training batch
        ray_batch = next(train_ray_iter)
        rays_o, rays_d = ray_batch['rays_o'][0].to(device), ray_batch['rays_d'][0].to(device)
        gt_rgb = ray_batch['gt_rgb'][0].to(device)
        render_out = volsdf_render_(rays_o, rays_d)

        rgb_loss = l1loss(gt_rgb, render_out['rgb']) # img2mse(gt_rgb, render_out['rgb']) # 
        eik_loss = eikonal_loss(render_out['grad_theta']) if 'grad_theta' in render_out else 0.
        
        loss = rgb_loss + args.eikonal_weight * eik_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr_sched.step()

        if i%args.i_print==0:
            mse = img2mse(gt_rgb, render_out['rgb'])
            psnr = mse2psnr(mse)
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()} (rgb: {rgb_loss.item()}, eik: {eik_loss.item()}, grad_theta: {render_out['grad_theta'].norm()})  PSNR: {psnr.item()}")
            if args.local_rank <= 0:
                tb_writer.add_scalar('loss', loss, i)
                tb_writer.add_scalar('psnr', psnr, i)
                if i%args.i_img==0:
                    # Log a rendered validation view to Tensorboard
                    img_i = torch.randint(len(test_raysampler), ())
                    ray_batch = test_raysampler[img_i]
                    rays_o, rays_d = ray_batch['rays_o'].to(device), ray_batch['rays_d'].to(device)
                    gt_rgb = ray_batch['gt_rgb'].to(device)
                    volsdf_render_.eval()
                    render_out = volsdf_render_(rays_o, rays_d)
                    volsdf_render_.train()
                    psnr = mse2psnr(img2mse(render_out['rgb'], gt_rgb))
                    tb_writer.add_scalar('psnr_eval', psnr, i)
                    H, W, _ = dataset.get_hwf()
                    tb_writer.add_image('gt_rgb', gt_rgb.reshape(H,W,-1), i, dataformats="HWC")
                    tb_writer.add_image('rgb', to8b(render_out['rgb'].cpu().numpy()).reshape(H,W,-1), i, dataformats='HWC')
                    if 'normal_map' in render_out:
                        normal_map = render_out['normal_map']
                        tb_writer.add_image('normal', normal_map.cpu().numpy().reshape(H,W,-1), i, dataformats='HWC')
                    
        if i%args.i_testset == 0 and i > 0 and args.local_rank <= 0:
            eval(volsdf_render_, test_rayloader, dataset.get_hwf()[:2], os.path.join(savedir, f'testset_{i:06d}'))
        
        if (i+1)%args.i_weights==0 and args.local_rank <= 0:
            path = os.path.join(savedir, f'{i+1:06d}.pt')
            torch.save({
                'global_step': i+1,
                'state_dict': volsdf_render.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)

def eval(volsdf_render, rayloader, img_hw, testsavedir):
    os.makedirs(testsavedir, exist_ok=True) 
    volsdf_render.eval()
    for i, ray_batch in enumerate(tqdm(rayloader)):
        rays_o, rays_d = ray_batch['rays_o'][0].to(device), ray_batch['rays_d'][0].to(device)
        gt_rgb = ray_batch['gt_rgb'][0].to(device)
        render_out = volsdf_render(rays_o, rays_d)
        psnr = mse2psnr(img2mse(gt_rgb, render_out['rgb']))
        imageio.imwrite(os.path.join(testsavedir, f'{i:03d}.png'), 
                    to8b(render_out['rgb'].cpu().numpy()).reshape(*img_hw,-1))
        tqdm.write(f"[Test] #: {i} PSNR: {psnr.item()}")
    volsdf_render.train()

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
