"""Only the four user-requested objectives; no additional constraints."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from monai.losses import DiceLoss

ARMS = ('dice', 'depth', 'depth_sdf3d', 'depth_sdf_layer')


def components(logits, labels, geometry=None, arm='dice') -> dict[str, torch.Tensor]:
    if arm not in ARMS:
        raise ValueError(arm)
    target = labels.long()
    if target.ndim == 4:
        target = target[:, None]
    if logits.shape[0] != 1 or logits.shape[1] != 3 or target.shape != logits[:, :1].shape:
        raise ValueError('Expected batch-one three-class logits and labels')
    logits = logits.float()
    values = {'dice': DiceLoss(to_onehot_y=True, softmax=True)(logits, target)}
    if arm == 'dice':
        return values
    weights = torch.as_tensor(geometry['weights'], device=logits.device, dtype=torch.float32)
    if weights.shape != logits.shape[2:]:
        raise ValueError('Geometry/logit shapes differ')
    values['depth'] = (F.cross_entropy(logits, target[:, 0], reduction='none')[0] * weights).sum()
    if arm in ('depth_sdf3d', 'depth_sdf_layer'):
        key = 'sdf_3d' if arm == 'depth_sdf3d' else 'sdf_layer'
        phi = torch.as_tensor(geometry[key], device=logits.device, dtype=torch.float32)
        truth = torch.stack((target[0, 0] == 1, target[0, 0] == 2)).float()
        values['sdf'] = (((logits.softmax(1)[0, 1:] - truth) * phi).mean(0) * weights).sum()
    return values


def combined_loss(values, coefficients, scale=1.):
    return values['dice'] + scale * sum(coefficients[name] * values[name]
                                       for name in values if name != 'dice')


def calibrated_coefficient(dice_rms, auxiliary_rms, target=.1, cap=.5) -> dict:
    """Training-only fixed coefficient: median ratio target and 95th-percentile cap."""
    base = np.asarray(dice_rms, dtype=float)
    aux = np.asarray(auxiliary_rms, dtype=float)
    if base.shape != aux.shape or base.size == 0 or not np.isfinite(base).all() or not np.isfinite(aux).all():
        raise ValueError('Invalid calibration observations')
    if (base <= 0).any() or (aux <= 0).any():
        raise ValueError('Zero calibration gradient; do not silently disable a requested term')
    ratios = aux / base
    value = min(target / np.median(ratios), cap / np.quantile(ratios, .95))
    return dict(coefficient=float(value), target_ratio=target, cap_ratio=cap,
                raw_ratio_median=float(np.median(ratios)), raw_ratio_p95=float(np.quantile(ratios, .95)),
                weighted_ratio_median=float(value * np.median(ratios)),
                weighted_ratio_p95=float(value * np.quantile(ratios, .95)), cases=len(base))
