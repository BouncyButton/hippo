"""Two separate changes to depth CE, with the original Dice objective."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from monai.losses import DiceLoss

ARMS = ('depth_class', 'depth_focal')
SETTINGS = dict(class_power=.5, class_count_floor=64, class_boost_cap=4., focal_gamma=1.)


def partial_class_weights(labels: np.ndarray, geometry: dict) -> np.ndarray:
    """Preserve every band's mass while partially balancing its three classes.

    Each class gets boost min(4, max(1, sqrt(N / max(n_class, 64)))).
    Renormalization within the band preserves its original depth weight.
    Counts use annotated labels inside the existing ROI only.
    """
    labels = np.asarray(labels)
    base = np.asarray(geometry['weights'], dtype=np.float64)
    bins = np.asarray(geometry['bin_ids'])
    if labels.shape != base.shape or bins.shape != labels.shape:
        raise ValueError('Geometry and label shapes differ')
    if not np.isin(labels, [0, 1, 2]).all():
        raise ValueError('Expected background, anterior, and posterior labels')
    if not np.isfinite(base).all() or (base < 0).any() or not np.isclose(base.sum(), 1.):
        raise ValueError('Invalid original depth weights')
    result = np.zeros_like(base)
    roi = base > 0
    for band in np.unique(bins[roi]):
        mask = roi & (bins == band)
        classes = labels[mask].astype(np.int64)
        counts = np.bincount(classes, minlength=3)
        boost = np.clip((len(classes) / np.maximum(counts, SETTINGS['class_count_floor']))
                        ** SETTINGS['class_power'], 1., SETTINGS['class_boost_cap'])
        weighted = base[mask] * boost[classes]
        result[mask] = weighted * (base[mask].sum() / weighted.sum())
    return result.astype(np.float32)


def components(logits, labels, geometry=None, arm='dice') -> dict[str, torch.Tensor]:
    if arm not in (*ARMS, 'dice'):
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
    key = 'class_weights' if arm == 'depth_class' else 'weights'
    weights = torch.as_tensor(geometry[key], device=logits.device, dtype=torch.float32)
    if weights.shape != logits.shape[2:]:
        raise ValueError('Geometry/logit shapes differ')
    ce = F.cross_entropy(logits, target[:, 0], reduction='none')[0]
    if arm == 'depth_focal':
        # Standard differentiable focal modulation, with no class alpha.
        # -expm1(-CE) evaluates 1-p_true accurately near p_true=1.
        ce = ce * (-torch.expm1(-ce)).pow(SETTINGS['focal_gamma'])
    values['depth'] = (ce * weights).sum()
    return values


def combined_loss(values, coefficients, scale=1.):
    return values['dice'] + scale * sum(coefficients[name] * values[name]
                                       for name in values if name != 'dice')


def calibrated_coefficient(dice_rms, auxiliary_rms, target=.1, cap=.5) -> dict:
    """Original training-only gradient calibration, unchanged between campaigns."""
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
