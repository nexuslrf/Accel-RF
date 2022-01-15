from typing import Dict, Tuple
import torch
from torch import Tensor

def nsvf_loss(
    r_out: Dict[str, Tensor], 
    gt_rgb: Tensor,
    loss_weights: Dict[str, float]) -> Tuple[Tensor, Dict[str, Tensor]]:
    losses = {}
    # computing loss
    if 'rgb' in loss_weights and loss_weights['rgb'] > 0:
        losses['rgb'] = ((r_out['rgb'] - gt_rgb)**2).mean() # remove .sum(-1), weights x3
    
    if 'alpha' in loss_weights and loss_weights['alpha'] > 0:
        _alpha = r_out['acc'].reshape(-1)
        losses['alpha'] = torch.log1p(1. / 0.11 * _alpha * (1 - _alpha)).mean()

    # if 'eikonal' in loss_weights and loss_weights['eikonal'] > 0: # Not supported..
    #     losses['eikonal'] = r_out['eikonal-term'].mean()
    
    if 'reg_term' in loss_weights and loss_weights['reg_term'] > 0:
        losses['reg_term'] = r_out['regz_term'].mean()

    loss = sum(losses[key] * loss_weights[key] for key in losses)
    
    return loss, losses