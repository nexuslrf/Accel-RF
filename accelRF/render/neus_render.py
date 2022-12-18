from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch import Tensor
from .nerf_render import volumetric_rendering
from .volsdf_render import VolSDFRender

class NeuSRender(VolSDFRender):
    def __init__(self, 
        embedder_pts: Optional[nn.Module], 
        embedder_views: Optional[nn.Module], 
        sdf_net: nn.Module, 
        rgb_net: nn.Module, 
        density_fn: nn.Module, 
        point_sampler: nn.Module, 
        bg_embedder_pts: Optional[nn.Module] = None, 
        bg_sdf_net: Optional[nn.Module] = None, 
        bg_rgb_net: Optional[nn.Module] = None, 
        bg_density_fn: Optional[nn.Module] = None, 
        scene_bounding_sphere: float = 3, 
        with_eikonal: bool = True, 
        white_bkgd: bool = False, 
        chunk: int = 1024 * 16,
        sample_eikonal: bool = False,
        anneal_end: int = 0,
    ):
        super().__init__(embedder_pts, embedder_views, sdf_net, rgb_net, density_fn, 
            point_sampler, bg_embedder_pts, bg_sdf_net, bg_rgb_net, bg_density_fn, 
            scene_bounding_sphere, with_eikonal, white_bkgd, chunk)

        self.sample_eikonal = sample_eikonal
        self.anneal_end = anneal_end
    
    def get_cos_anneal_ratio(self, iters):
        if self.anneal_end == 0.0 or iters < 0:
            return 1.0
        else:
            return min(1.0, iters / self.anneal_end)

    def forward(self, rays_o: Tensor, rays_d: Tensor, iters=-1):
        '''
        Args:
            rays_o: Tensor, sampled ray origins, [N_rays, 3]
            rays_d: Tensor, sampled ray directions, [N_rays, 3], normalized.
        '''
        cos_anneal_ratio = self.get_cos_anneal_ratio(iters)
        N_rays = rays_o.shape[0]
        if N_rays > self.chunk:
            ret = [self.forward(rays_o[ci:ci+self.chunk], rays_d[ci:ci+self.chunk]) for ci in range(0, N_rays, self.chunk)]
            ret = {
                k: torch.cat([out[k] for out in ret], 0)
                for k in ret[0]
            }
            return ret
        else:
            ret = {}
            # start point sampling
            sample_out = self.point_sampler(rays_o, rays_d, self.sdf_net, 
                                        self.density_fn, self.embedder_pts)
            pts, z_vals, pts_bg, z_vals_bg, pts_eik_near = sample_out
            # dist
            dists = z_vals[...,1:] - z_vals[...,:-1]
            # mid points
            pts = 0.5 * (pts[...,1:,:] + pts[..., :-1,:])

            N_samples = pts.shape[-2]
            pts.requires_grad_(True)
            pts_embed = self.embedder_pts(pts) # [N_rays, N_samples, pe_dim]
            view_embed = self.embedder_views(rays_d[...,None,:]) # [N_rays, 1, ve_dim]
            sdf, feats = self.sdf_net(pts_embed, pts)
            gradients = torch.autograd.grad(sdf, pts, torch.ones_like(sdf), 
                                retain_graph=self.training, create_graph=self.training)[0]
            with torch.set_grad_enabled(self.training):
                rgb = self.rgb_net(view_embed.expand(*pts.shape[:-1], -1), feats, pts, gradients)
                alpha = self.density_fn(sdf, rays_d[...,None,:], dists, gradients, cos_anneal_ratio)
                white_bkgd = True if self.white_bkgd else False
                r_ret = volumetric_rendering(rgb, alpha[...,None], z_vals, white_bkgd=white_bkgd, 
                                    rgb_only=True, with_alpha=True, with_T=True)
                ret['rgb'] = r_ret['rgb']    
                if self.inverse_sphere_bg:
                    # Background rendering
                    pts_embed_bg = self.bg_embedder_pts(pts_bg)
                    sdf_bg, feats_bg = self.bg_sdf_net(pts_embed_bg, pts_bg)
                    rgb_bg = self.bg_rgb_net(view_embed.expand(*pts_bg.shape[:-1], -1), feats_bg)
                    density_bg = self.bg_density_fn(sdf_bg)
                    r_ret_bg = volumetric_rendering(rgb_bg, density_bg, z_vals_bg, white_bkgd=False, rgb_only=True)
                    # Composite foreground and background
                    ret['rgb'] = r_ret['transmittance'][...,-1:] * r_ret_bg['rgb'] + r_ret['rgb']

                if self.training:
                    if self.sample_eikonal:
                        # Sample points for the eikonal loss
                        n_eik_points = N_rays
                        pts_eik = torch.empty(n_eik_points, 3, device=rays_o.device).uniform_(-self.scene_bounding_sphere, self.scene_bounding_sphere)
                        pts_eik = torch.cat([pts_eik, pts_eik_near.reshape(-1,3)], 0) if pts_eik_near is not None else pts_eik

                        pts_eik.requires_grad_(True)
                        sdf_eik, _ = self.sdf_net(self.embedder_pts(pts_eik), pts_eik, sdf_only=True)
                        grad_theta = torch.autograd.grad(sdf_eik, pts_eik, torch.ones_like(sdf_eik), retain_graph=True, create_graph=True)[0]
                        ret['grad_theta'] = grad_theta
                    else:
                        ret['grad_theta'] = gradients

                else:
                    gradients = gradients.detach()
                    normals = gradients / gradients.norm(2, -1, keepdim=True)
                    normals = normals.reshape(-1, N_samples, 3)
                    normal_map = torch.sum(r_ret['weights'].unsqueeze(-1) * normals, 1)
                    normal_map = r_ret['transmittance'][...,-1:] * 1 + normal_map
                    ret['normal_map'] = (normal_map + 1) / 2.

            return ret