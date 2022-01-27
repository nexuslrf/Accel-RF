import torch
import torch.nn as nn
from torch import Tensor
from .nerf_pointsampler import get_z_vals, uniform_sample, cdf_sample

@torch.jit.script
def get_sphere_intersections(rays_o: Tensor, rays_d: Tensor, r: float=1.0):
    '''
    Ray-sphere Intersection: https://en.wikipedia.org/wiki/Line-sphere_intersection
    Input: n_rays x 3 ; n_rays x 3
    Output: n_rays x 1, n_rays x 1 (close and far)
    '''
    ray_cam_dot = torch.bmm(rays_d.view(-1, 1, 3),
                            rays_o.view(-1, 3, 1)).squeeze(-1)
    under_sqrt = ray_cam_dot ** 2 - (rays_o.norm(2, 1, keepdim=True) ** 2 - r ** 2)

    sphere_intersections = torch.sqrt(under_sqrt) * torch.tensor([-1., 1.], device=rays_o.device) - ray_cam_dot
    sphere_intersections = sphere_intersections.clamp_min(0.0)

    return sphere_intersections # [t0, t1]

def get_error_bound(
        beta: Tensor, density_fn: nn.Module, sdf: Tensor, dists: Tensor, d_star: Tensor
    ) -> Tensor:
    density = density_fn(sdf, beta=beta)
    shifted_free_energy = torch.cat([torch.zeros(dists.shape[0], 1).cuda(), dists * density[:, :-1]], dim=-1)
    integral_estimation = torch.cumsum(shifted_free_energy, dim=-1)
    error_per_section = torch.exp(-d_star / beta) * (dists ** 2.) / (4 * beta ** 2)
    error_integral = torch.cumsum(error_per_section, dim=-1)
    bound_opacity = (torch.clamp(torch.exp(error_integral), max=1.e6) - 1.0) * torch.exp(-integral_estimation[:, :-1])

    return bound_opacity.max(-1)[0]

@torch.jit.script # this loop body is a good candidate for JIT.
def error_bound_sampling_update(
        z_vals, density_fn, sdf, beta0, beta, eps, beta_iters,
        last_iter, add_tiny, N_samples, N_samples_eval, det
    ):
    device = z_vals.device
    # Calculating the bound d* (Theorem 1)
    d = sdf # [N, S]
    dists = z_vals[:, 1:] - z_vals[:, :-1]
    a, b, c = dists, d[:, :-1].abs(), d[:, 1:].abs()
    first_cond = a.pow(2) + b.pow(2) <= c.pow(2)
    second_cond = a.pow(2) + c.pow(2) <= b.pow(2)
    d_star = torch.zeros(z_vals.shape[0], z_vals.shape[1] - 1, device=device)
    d_star[first_cond] = b[first_cond]
    d_star[second_cond] = c[second_cond]
    s = (a + b + c) / 2.0
    area_before_sqrt = s * (s - a) * (s - b) * (s - c)
    mask = ~first_cond & ~second_cond & (b + c - a > 0)
    d_star[mask] = (2.0 * torch.sqrt(area_before_sqrt[mask])) / (a[mask])
    d_star = (d[:, 1:].sign() * d[:, :-1].sign() == 1) * d_star  # Fixing the sign

    # Updating beta using line search
    curr_error = get_error_bound(beta0, density_fn, sdf, dists, d_star)
    beta[curr_error <= eps] = beta0
    beta_min, beta_max = beta0.unsqueeze(0).repeat(z_vals.shape[0]), beta
    for j in range(beta_iters):
        beta_mid = (beta_min + beta_max) / 2.
        curr_error = get_error_bound(beta_mid.unsqueeze(-1), density_fn, sdf, dists, d_star)
        beta_max[curr_error <= eps] = beta_mid[curr_error <= eps]
        beta_min[curr_error > eps] = beta_mid[curr_error > eps]
    beta = beta_max

    # Upsample more points
    density = density_fn(sdf, beta=beta.unsqueeze(-1))

    dists = torch.cat([dists, 1e10*torch.ones(dists.shape[0],1, device=device)], -1)
    free_energy = dists * density
    shifted_free_energy = torch.cat([torch.zeros(dists.shape[0], 1, device=device), free_energy[:, :-1]], dim=-1)
    alpha = 1 - torch.exp(-free_energy)
    transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))

    #  Check if we are done and this is the last sampling
    not_converge = beta.max() > beta0
    add_more = not_converge and not last_iter

    if add_more:
        ''' Sample more points proportional to the current error bound'''
        N = N_samples_eval
        error_per_section = torch.exp(-d_star / beta.unsqueeze(-1)) * (dists ** 2.) / (4 * beta.unsqueeze(-1) ** 2)
        error_integral = torch.cumsum(error_per_section, dim=-1)
        bound_opacity = (torch.clamp(torch.exp(error_integral),max=1.e6) - 1.0) * transmittance
        weights = bound_opacity
        cdf_eps = add_tiny
    else:
        ''' Sample the final sample set to be used in the volume rendering integral '''
        N = N_samples
        weights = alpha * transmittance  # probability of the ray hits something here
        cdf_eps = 1e-5
    
    cdf_det = add_more or det
    samples = cdf_sample(N, z_vals, weights, cdf_det, mid_bins=False, eps=cdf_eps, include_init_z_vals=False)
    # Adding samples if we not converged
    samples_idx = None
    if add_more:
        z_vals, samples_idx = torch.sort(torch.cat([z_vals, samples], -1), -1)
    return samples, z_vals, samples_idx, beta, not_converge

class VolSDFPointSampler(nn.Module):
    '''
    Basically equivalent to VolSDF's ErrorBoundSampler
    https://github.com/lioryariv/volsdf/blob/main/code/model/ray_sampler.py#L46
    '''
    def __init__(self, scene_bounding_sphere, near, N_samples, N_samples_eval, N_samples_extra,
                 eps, beta_iters, max_total_iters, inverse_sphere_bg=False, 
                 N_samples_inverse_sphere=0, add_tiny=0.0, with_eik_sample=True):
        super().__init__()
        self.near, self.far = near, 2*scene_bounding_sphere
        self.N_samples = N_samples
        self.N_samples_eval = N_samples_eval
        self.N_samples_extra = N_samples_extra
        self.N_samples_inverse_sphere = N_samples_inverse_sphere

        self.eps = eps
        self.beta_iters = beta_iters
        self.max_total_iters = max_total_iters
        self.scene_bounding_sphere = scene_bounding_sphere
        self.add_tiny = add_tiny

        self.inverse_sphere_bg = inverse_sphere_bg
        self.with_eik_sample = with_eik_sample

    def forward(
        self, rays_o: Tensor, rays_d: Tensor, sdf_net: nn.Module, 
        density_fn: nn.Module, pts_embedder: nn.Module
    ):
        device = rays_o.device
        beta0 = density_fn.get_beta().detach()
        _ones = torch.ones(rays_o.shape[0], 1, device=device)
        near = self.near * _ones

        if not self.inverse_sphere_bg:
            far = self.far * _ones
        else:
            far = get_sphere_intersections(rays_o, rays_d, self.scene_bounding_sphere)[:,1:]

        # Start with uniform sampling
        init_z_vals = get_z_vals(near, far, self.N_samples_eval, device=device)
        z_vals = uniform_sample(self.N_samples_eval, rays_d.shape[0], init_z_vals=init_z_vals, 
                            perturb=1 if sdf_net.training else 0, device=rays_d.device)
        samples, samples_idx = z_vals, None

        # Get maximum beta from the upper bound (Lemma 2)
        dists = z_vals[:, 1:] - z_vals[:, :-1]
        bound = (1.0 / (4.0 * torch.log(torch.tensor(self.eps + 1.0)))) * (dists ** 2.).sum(-1)
        beta = torch.sqrt(bound)

        total_iters, not_converge = 0, True

        # Algorithm 1
        while not_converge and total_iters < self.max_total_iters:
            points = rays_o.unsqueeze(1) + samples.unsqueeze(2) * rays_d.unsqueeze(1)
            points_flat = points.reshape(-1, 3)

            # Calculating the SDF only for the new sampled points
            with torch.no_grad():
                samples_sdf, _ = sdf_net(pts_embedder(points_flat), points_flat, sdf_only=True)
            if samples_idx is not None:
                sdf_merge = torch.cat([sdf.reshape(-1, z_vals.shape[1] - samples.shape[1]),
                                       samples_sdf.reshape(-1, samples.shape[1])], -1)
                sdf = torch.gather(sdf_merge, 1, samples_idx).reshape(-1, 1)
            else:
                sdf = samples_sdf

            sdf = sdf.reshape_as(z_vals) # [N_rays, N_samples]
            total_iters += 1
            last_iter = total_iters == self.max_total_iters
            samples, z_vals, samples_idx, beta, not_converge = error_bound_sampling_update(
                z_vals, density_fn, sdf, beta0, beta, self.eps, self.beta_iters, last_iter,
                self.add_tiny, self.N_samples, self.N_samples_eval, (not sdf_net.training)
            )
            
        z_samples = samples

        if self.N_samples_extra > 0:
            if sdf_net.training:
                sampling_idx = torch.randperm(z_vals.shape[1])[:self.N_samples_extra]
            else:
                sampling_idx = torch.linspace(0, z_vals.shape[1]-1, self.N_samples_extra).long()
            z_vals_extra = torch.cat([near, far, z_vals[:,sampling_idx]], -1)
        else:
            z_vals_extra = torch.cat([near, far], -1)

        z_vals, _ = torch.sort(torch.cat([z_samples, z_vals_extra], -1), -1)
        
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,None] # [N_rays, N_samples, 3]

        # add some of the near surface points
        pts_eik = None
        if self.with_eik_sample and self.training:
            idx = torch.randint(z_vals.shape[-1], (z_vals.shape[0],)).cuda()
            z_samples_eik = torch.gather(z_vals, 1, idx.unsqueeze(-1))
            pts_eik = rays_o[...,None,:] + rays_d[...,None,:] * z_samples_eik[...,None]

        pts_bg, z_vals_bg = None, None
        if self.inverse_sphere_bg:
            z_vals_inverse_sphere = uniform_sample(self.N_samples_inverse_sphere, rays_d.shape[0],
                                            perturb=1 if sdf_net.training else 0, device=rays_d.device)
            z_vals_inverse_sphere = z_vals_inverse_sphere * (1./self.scene_bounding_sphere)
            z_vals_bg = torch.flip(z_vals_inverse_sphere, dims=[-1,])
            pts_bg = depth2pts_outside(rays_o, rays_d, z_vals_bg, self.scene_bounding_sphere)

        return pts, z_vals, pts_bg, z_vals_bg, pts_eik


@torch.jit.script
def depth2pts_outside(ray_o, ray_d, depth, bounding_sphere):
    '''
    ray_o, ray_d: [..., 3]
    depth: [...]; inverse of distance to sphere origin
    '''
    o_dot_d = torch.sum(ray_d * ray_o, dim=-1)
    under_sqrt = o_dot_d ** 2 - ((ray_o ** 2).sum(-1) - bounding_sphere ** 2)
    d_sphere = torch.sqrt(under_sqrt) - o_dot_d
    p_sphere = ray_o + d_sphere.unsqueeze(-1) * ray_d
    p_mid = ray_o - o_dot_d.unsqueeze(-1) * ray_d
    p_mid_norm = torch.norm(p_mid, dim=-1)

    rot_axis = torch.cross(ray_o, p_sphere, dim=-1)
    rot_axis = rot_axis / torch.norm(rot_axis, dim=-1, keepdim=True)
    phi = torch.asin(p_mid_norm / bounding_sphere)
    theta = torch.asin(p_mid_norm * depth)  # depth is inside [0, 1]
    rot_angle = (phi - theta).unsqueeze(-1)  # [..., 1]

    # now rotate p_sphere
    # Rodrigues formula: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    p_sphere_new = p_sphere * torch.cos(rot_angle) + \
                    torch.cross(rot_axis, p_sphere, dim=-1) * torch.sin(rot_angle) + \
                    rot_axis * torch.sum(rot_axis * p_sphere, dim=-1, keepdim=True) * (1. - torch.cos(rot_angle))
    p_sphere_new = p_sphere_new / torch.norm(p_sphere_new, dim=-1, keepdim=True)
    pts = torch.cat((p_sphere_new, depth.unsqueeze(-1)), dim=-1)

    return pts