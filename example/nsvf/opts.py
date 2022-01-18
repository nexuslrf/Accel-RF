import configargparse

def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', 
                        help='input data directory')
    parser.add_argument("--scene", type=str, default='lego')

    # sampling options
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--N_views", type=int, default=1, 
                        help='number of view used in PerViewRaySampler')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--full_rays", action='store_true', help='used for PerViewRaySampler')
    # training options
    parser.add_argument("--N_iters", type=int, default=200000,
                        help='the number of training iterations')    
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights file to reload for network')
    
    # model options
    parser.add_argument("--D_feat", type=int, default=3, 
                        help='layers in network')
    parser.add_argument("--W_feat", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--D_sigma", type=int, default=2)
    parser.add_argument("--W_sigma", type=int, default=128)
    parser.add_argument("--D_rgb", type=int, default=5)
    parser.add_argument("--W_rgb", type=int, default=256)
    parser.add_argument("--no_layernorm", action='store_true')

    parser.add_argument("--multires", type=int, default=6, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    # nsvf options
    parser.add_argument("--voxel_size", type=float, default=0.4)
    parser.add_argument("--step_size", type=float, default=0.125)
    parser.add_argument("--embed_dim", type=int, default=32)
    parser.add_argument("--loss_w_rgb", type=float, default=256)
    parser.add_argument("--loss_w_alpha", type=float, default=1)
    parser.add_argument("--loss_w_reg", type=float, default=0)
    parser.add_argument("--pruning_thres", type=float, default=0.5)
    parser.add_argument("--pruning_every_steps", type=int, default=0)
    parser.add_argument("--pruning_steps", nargs='+', type=int, default=[])
    parser.add_argument("--splitting_steps", nargs='+', type=int, default=[])
    parser.add_argument("--half_stepsize_steps", nargs='+', type=int, default=[])
    parser.add_argument("--bg_field", action='store_true')
    parser.add_argument("--min_color", type=int, default=-1)
    parser.add_argument("--early_stop_thres", type=float, default=0, 
                        help="values in the form of $T$, where $T = \exp(-\sum_i\sigma_i\delta_i)$")

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')

    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 
    # distributed options
    parser.add_argument("--local_rank", type=int, default=-1, 
                        help='node rank for distributed training')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', 
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=1000, 
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000, 
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000, 
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=50000, 
                        help='frequency of render_poses video saving')

    return parser