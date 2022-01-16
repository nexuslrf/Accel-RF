from typing import Callable, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from accelRF.rep import Explicit3D
from accelRF.render.nerf_render import volumetric_rendering
from accelRF.rep.utils import offset_points

def masked_scatter(mask: torch.BoolTensor, x: Tensor):
    B, K = mask.size()
    if x.dim() == 1:
        return x.new_zeros(B, K).masked_scatter(mask, x)
    return x.new_zeros(B, K, x.size(-1)).masked_scatter(
        mask.unsqueeze(-1).expand(B, K, x.size(-1)), x)

def fill_in(shape, hits: Tensor, input: Tensor, initial=0.):
    if torch.Size(shape) == input.shape:
        return input   # shape is the same no need to fill

    if isinstance(initial, torch.Tensor):
        output = initial.expand(*shape)
    else:
        output = input.new_ones(*shape) * initial
    if input is not None:
        if len(shape) == 1:
            output = output.masked_scatter(hits, input)
        output = output.masked_scatter(hits.unsqueeze(-1).expand(*shape), input)
    return output

class NSVFRender(nn.Module):
    def __init__(
        self,
        pts_embedder: Optional[nn.Module],
        view_embedder: Optional[nn.Module],
        voxel_embedder: nn.Module,
        model: nn.Module,
        point_sampler: nn.Module,
        vox_rep: Explicit3D,
        early_stop_thres: float=0,
        bg_color: Optional[nn.Module]=None,
        white_bkgd: bool=False,
        fwd_steps: int=4,
        chunk: int=1024*64
    ):
        '''
        Args:
            early_stop_thres: values in the form of $T$, where $T = \exp(-\sum_i\sigma_i\delta_i)$
        '''
        super().__init__()
        self.pts_embedder = pts_embedder if pts_embedder is not None else nn.Identity()
        self.view_embedder = view_embedder if view_embedder is not None else nn.Identity()
        self.voxel_embedder = voxel_embedder
        self.model = model
        self.point_sampler = point_sampler
        self.vox_rep = vox_rep
        self.early_stop = (early_stop_thres > 0)
        self.early_stop_thres = -np.log(early_stop_thres) if self.early_stop else 0
        self.chunk = chunk
        self.fwd_steps = fwd_steps
        self.bg_color = bg_color
        self.white_bkgd = (white_bkgd and bg_color is None)


    def forward(self, rays_o: Tensor, rays_d: Tensor, vox_idx: Tensor, 
        t_near: Tensor, t_far: Tensor, ray_hits: Tensor):
        '''
        Args:
            rays_o, rays_d: [N_rays, 3]
            vox_idx, t_near, t_far: [N_rays, max_hits]
            hits: [N_rays]
        '''
        n_rays = ray_hits.shape[0]
        if n_rays > self.chunk:
            ret = [self.forward(rays_o[ci:ci+self.chunk], rays_d[ci:ci+self.chunk]) 
                        for ci in range(0, n_rays, self.chunk)]
            ret = {
                k: torch.cat([out[k] for out in ret], 0)
                for k in ret[0]
            }
            return ret
        else:
            if ray_hits.sum() > 0:
                vox_idx, t_near, t_far = vox_idx[ray_hits], t_near[ray_hits], t_far[ray_hits] # [n_hits, max_hits]
                rays_o, rays_d = rays_o[ray_hits], rays_d[ray_hits] # [n_hits, 3]
                # point sampling
                sample_out = self.point_sampler(rays_o, rays_d, vox_idx, t_near, t_far)
                pts, p2v_idx, t_vals, dists = sample_out # [n_hits, max_samples] + [3], [], []
                mask_pts = p2v_idx.ne(-1)
                max_samples= pts.shape[1]
                n_pts_fwd, start_step = 0, 0
                accum_log_T = 0
                nn_out = []
                for i in range(0, max_samples, self.fwd_steps):
                    n_pts_step = mask_pts[:, i:i+self.fwd_steps].sum()
                    if (i+self.fwd_steps >= max_samples) or (n_pts_fwd + n_pts_step > self.chunk):
                        l, r = start_step, i+self.fwd_steps
                        out, n_pts = self.step_forward(
                            pts[:,l:r], p2v_idx[:,l:r], rays_d, mask_pts[:,l:r], dists[:,l:r])
                        nn_out.append(out)
                        start_step, n_pts_fwd = r, 0
                        # oops, early_stop relies on free_energy (sigma * dist)
                        if self.early_stop:
                            with torch.no_grad():
                                accum_log_T = accum_log_T + out['free_energy'].sum(1)
                                mask_early_stop = accum_log_T > self.early_stop_thres
                                mask_pts[mask_early_stop] = 0
                    else:
                        n_pts_fwd += n_pts_step
                nn_out = {
                    k: torch.cat([out[k] for out in nn_out], 1) for k in nn_out[0]
                }
                r_out = volumetric_rendering(nn_out['rgb'], nn_out['free_energy'], t_vals, 
                            white_bkgd=self.white_bkgd, with_dist=True)   
                ret = {'rgb': r_out['rgb'], 'disp': r_out['disp'], 'acc': r_out['acc']}
                if 'feat_n2' in nn_out:
                    ret['regz_term'] = (nn_out['feat_n2'] * r_out['weights']).sum(-1)
            else:
                ret = {'rgb': None, 'disp': None, 'acc': None}
            # TODO post-processing...
            # fill_in un-hitted holes
            ret['rgb'] = fill_in((n_rays, 3), ray_hits, ret['rgb'], 0.0)
            ret['disp'] = fill_in((n_rays,), ray_hits, ret['disp'], 0.0)
            ret['acc'] = fill_in((n_rays, ), ray_hits, ret['acc'], 0.0)
            if self.bg_color is not None:
                bg_rgb = self.bg_color(ret['rgb'])
                missed = (1 - ret['acc'])
                ret['rgb'] = ret['rgb'] + missed[...,None] * bg_rgb
            
            return ret

    def step_forward(self, pts: Tensor, p2v_idx: Tensor, rays_d: Tensor, 
        mask_pts: Tensor, dists: Optional[Tensor]=None):
        n_pts = mask_pts.sum()
        if n_pts == 0:
            return None, 0

        rays_d = rays_d[...,None,:].expand_as(pts)[mask_pts] # [n_pts, 3]
        pts, p2v_idx = pts[mask_pts], p2v_idx[mask_pts]
        
        vox_embeds = self.voxel_embedder(pts, p2v_idx, self.vox_rep)
        nn_out = self.model(
            self.pts_embedder(vox_embeds), self.view_embedder(rays_d)
        )

        # scatter back...
        out = {'rgb': masked_scatter(mask_pts, nn_out['rgb'])}
        if dists is not None:
            dists = dists[mask_pts]
            free_energy = F.relu(nn_out['sigma']) * dists # activation is here!
            out['free_energy'] = masked_scatter(mask_pts, free_energy)
        else:
            out['sigma'] = masked_scatter(mask_pts, nn_out['sigma'])
        if 'feat_n2' in nn_out:
            out['feat_n2'] = masked_scatter(mask_pts, nn_out['feat_n2'])
        return out, n_pts

    @torch.no_grad()
    def pruning(self, thres=0.5, bits=16):
        '''
        Based on NSVF's pruning function. fairnr/modules/encoder.py#L606
        '''
        center_pts = self.vox_rep.center_points # [n_vox, 3]
        device = center_pts.device
        n_vox, n_pts_vox = center_pts.shape[0], bits**3 # sample bits^3 points per voxel
        rel_pts = offset_points(bits, scale=0, device=device) # [bits**3, 3], scale [0,1]
        p2v_idx = torch.arange(n_vox, device=device) # [n_vox]
        # get pruning scores
        vox_chunk = (self.chunk - 1) // n_pts_vox + 1
        scores = []
        for i in range(0, n_vox, vox_chunk):
            vox_embeds = self.voxel_embedder(rel_pts, p2v_idx[i:i+vox_chunk], 
                                self.vox_rep, per_voxel=True) # [vc, bits**3, emb_dim]
            sigma = self.model(self.pts_embedder(vox_embeds))['sigma'][...,0] # [vc, bits**3]
            scores.append(torch.exp(-sigma.relu()).min(-1)[0]) # [vc]
        scores = torch.cat(scores, 0) 
        keep = (1 - scores) > thres
        emb_idx = self.vox_rep.pruning(keep)
        if emb_idx is not None:
            new_embeddings = self.voxel_embedder.embeddings.weights[emb_idx]
            self.voxel_embedder.update_embeddings(new_embeddings)
        print(f"pruning done. # of voxels before: {keep.size(0)}, after: {keep.sum()} voxels")

    @torch.no_grad()
    def splitting(self):
        embeddings = self.voxel_embedder.embeddings.weights
        new_embeddings = self.vox_rep.splitting(embeddings)
        if new_embeddings is not None:
            self.voxel_embedder.update_embeddings(new_embeddings)
        
    @torch.no_grad()
    def half_stepsize(self):
        self.point_sampler.half_stepsize()