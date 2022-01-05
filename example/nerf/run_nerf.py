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

from accelRF.datasets import Blender, LLFF
from accelRF.raysampler import NeRFRaySampler
from accelRF.pointsampler import NeRFPointSampler
from accelRF.models import PositionalEncoding, NeRF
from accelRF.render.nerf_render import NeRFRender

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
# def metrics TODO metrics can be pre-defined in a accelRF collection
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = torch.no_grad()(lambda x : -10. * torch.log10(x))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

def main():
    # prepare data
    if args.dataset_type == 'blender':
        dataset = Blender(args.datadir, args.scene, args.half_res, args.testskip, args.white_bkgd)
    elif args.dataset_type == 'llff':
        dataset = LLFF(args.datadir, args.scene, args.factor, spherify=args.spherify, 
            use_ndc=(not args.no_ndc), n_holdout=args.llffhold)
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
        white_bkgd=args.white_bkgd,
        chunk=args.chunk * n_gpus,
        use_ndc=(not args.no_ndc), hwf=dataset.get_hwf()
    )
    if args.local_rank >=0:
        nerf_render = torch.nn.parallel.DistributedDataParallel(nerf_render.to(device), device_ids=[args.local_rank])
    else:
        nerf_render = nn.DataParallel(nerf_render.jit_script()).to(device)
    # create optimizer
    optimizer  = optim.Adam(nerf_render.parameters(), args.lrate, betas=(0.9, 0.999))
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
        except:
            pass
        nerf_render.load_state_dict(ckpt['state_dict'])

    lr_sched = optim.lr_scheduler.ExponentialLR(optimizer, 0.1**(1/(args.lrate_decay*1000)), last_epoch=start-1)

    # prepare dataloader
    train_raysampler = NeRFRaySampler(
        dataset.get_sub_set('train'), args.N_rand, args.N_iters, use_batching=(not args.no_batching), 
        precrop=(args.precrop_iters > 0), precrop_frac=args.precrop_frac, precrop_iters=args.precrop_iters, 
        start_epoch=start, rank=args.local_rank, n_replica=n_replica) # ~~use_ndc=(not args.no_ndc)~~
    test_raysampler = NeRFRaySampler(dataset.get_sub_set('test'), full_rendering=True)
    val_raysampler = NeRFRaySampler(dataset.get_sub_set('val'), full_rendering=True)
    train_rayloader = DataLoader(train_raysampler, num_workers=1, pin_memory=True)
    test_rayloader = DataLoader(test_raysampler, num_workers=1, pin_memory=True)

    train_ray_iter = iter(train_rayloader)
    for i in trange(start, args.N_iters):
        # get one training batch
        ray_batch = next(train_ray_iter)
        rays_o, rays_d = ray_batch['rays_o'][0].to(device), ray_batch['rays_d'][0].to(device)
        gt_rgb = ray_batch['gt_rgb'][0].to(device)
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

        if i%args.i_print==0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")
            if args.local_rank <= 0:
                tb_writer.add_scalar('loss', loss, i)
                tb_writer.add_scalar('psnr', psnr, i)
                if 'rgb0' in render_out:
                    tb_writer.add_scalar('psnr0', psnr0, i)
                if i%args.i_img==0:
                    # Log a rendered validation view to Tensorboard
                    img_i=torch.randint(len(val_raysampler), ())
                    ray_batch = val_raysampler[img_i]
                    rays_o, rays_d = ray_batch['rays_o'].to(device), ray_batch['rays_d'].to(device)
                    gt_rgb = ray_batch['gt_rgb'].to(device)
                    with torch.no_grad():
                        render_out = nerf_render(rays_o, rays_d)
                    psnr = mse2psnr(img2mse(render_out['rgb'], gt_rgb))
                    tb_writer.add_scalar('psnr_eval', psnr, i)
                    H, W, _ = dataset.get_hwf()
                    tb_writer.add_image('gt_rgb', gt_rgb.reshape(H,W,-1), i, dataformats="HWC")
                    tb_writer.add_image('rgb', to8b(render_out['rgb'].cpu().numpy()).reshape(H,W,-1), i, dataformats='HWC')
                    tb_writer.add_image('disp', render_out['disp'].reshape(H,W), i, dataformats="HW")
                    tb_writer.add_image('acc', render_out['acc'].reshape(H,W), i, dataformats="HW")
                    if 'rgb0' in render_out:
                        tb_writer.add_image('rgb0', to8b(render_out['rgb0'].cpu().numpy()).reshape(H,W,-1), i, dataformats='HWC')
                        tb_writer.add_image('disp0', render_out['disp0'].reshape(H,W), i, dataformats="HW")
                    
        if i%args.i_testset == 0 and i > 0:
            eval(nerf_render, test_rayloader, dataset.get_hwf()[:2], os.path.join(savedir, f'testset_{i:06d}'))
        
        if (i+1)%args.i_weights==0 and args.local_rank <= 0:
            path = os.path.join(savedir, f'{i+1:06d}.pt')
            torch.save({
                'global_step': i+1,
                'state_dict': nerf_render.state_dict(), # keys start with 'module'
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
        
def eval(nerf_render, rayloader, img_hw, testsavedir):
    os.makedirs(testsavedir, exist_ok=True) 
    nerf_render.eval()
    for i, ray_batch in enumerate(tqdm(rayloader)):
        rays_o, rays_d = ray_batch['rays_o'][0].to(device), ray_batch['rays_d'][0].to(device)
        gt_rgb = ray_batch['gt_rgb'][0].to(device)
        with torch.no_grad():
            render_out = nerf_render(rays_o, rays_d)
        psnr = mse2psnr(img2mse(gt_rgb, render_out['rgb']))
        imageio.imwrite(os.path.join(testsavedir, f'{i:03d}.png'), 
                    to8b(render_out['rgb'].cpu().numpy()).reshape(*img_hw,-1))
        tqdm.write(f"[Test] #: {i} PSNR: {psnr.item()}")
    nerf_render.train()

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
