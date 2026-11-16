"""HFH-only conditional residual-band ablation; canonical BAND is unchanged.

All decisions use the current training logits and that case's annotation.
Calibration uses a fixed-normalization envelope containing every possible
selected voxel, so the actual logit-gradient norm is bounded at the anchor.
That bound is not a guarantee after the model changes.
"""
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib

import numpy as np
from scipy import ndimage as ndi
import torch
import torch.nn.functional as F

from thesis.new_constraints.constraint_result import ConstraintResult, differentiable_zero


@dataclass(frozen=True)
class Policy:
    spacing_mm: tuple = (1., 2., 1.)
    widths_mm: tuple = (1., 2., 3.)
    minimum_training_dice: float = .5
    minimum_error_coverage: float = .6
    maximum_inner_foreground_fraction: float = .75
    minimum_side_errors: int = 8


POLICY = Policy()
VERSION = 'hfh_nnunet_conditional_residual_band_v1'
EXTRA_UPDATE_FLOPS = 256 * 64**3 + 65536


@lru_cache(maxsize=64)
def geometry(annotation: bytes):
    """Static label preprocessing; never consumes model or validation errors."""
    truth = np.frombuffer(annotation, dtype=np.uint8).reshape(64, 64, 64).astype(bool)
    roi = np.zeros_like(truth); roi[:, 16:48, :] = True
    if not truth.any() or np.any(truth & ~roi):
        raise ValueError('Expected a nonempty HFH annotation inside the unpadded ROI')
    # Same half-minimum-spacing discrete EDT convention as the project SDF.
    # These are voxel-center distance estimates, not exact surface distances.
    inner_distance = np.maximum(ndi.distance_transform_edt(truth, sampling=POLICY.spacing_mm)-.5, 0)
    outer_distance = np.maximum(ndi.distance_transform_edt(~truth, sampling=POLICY.spacing_mm)-.5, 0)
    inner = np.stack([truth & (inner_distance <= w) for w in POLICY.widths_mm])
    outer = np.stack([~truth & roi & (outer_distance <= w) for w in POLICY.widths_mm])
    return truth, roi, inner, outer


def selection(truth, prediction, roi, inner, outer):
    """Choose each boundary side independently; return detached support and reasons."""
    fp, fn = prediction & ~truth & roi, ~prediction & truth
    tp, count = int((prediction & truth).sum()), int(truth.sum())
    fp_count, fn_count = int(fp.sum()), int(fn.sum())
    dice = 2*tp/max(1, 2*tp+fp_count+fn_count)
    selected = np.zeros_like(truth)
    report = dict(training_dice=dice, tp=tp, fp=fp_count, fn=fn_count,
                  foreground_voxels=count, ready=dice >= POLICY.minimum_training_dice, sides={})
    for side, errors, shells in (('inner', fn, inner), ('outer', fp, outer)):
        total = int(errors.sum())
        measurements = [dict(width_mm=w, shell_voxels=int(shell.sum()),
            covered_errors=int((errors & shell).sum()),
            error_coverage=int((errors & shell).sum())/total if total else 0.)
            for w, shell in zip(POLICY.widths_mm, shells)]
        entry = dict(total_errors=total, candidates=measurements, selected_width_mm=None,
                     selected_errors=0, reason='no_suitable_local_shell')
        if not report['ready']:
            entry['reason'] = 'foreground_not_ready'
        elif total < POLICY.minimum_side_errors:
            entry['reason'] = 'too_few_errors'
        else:
            for shell, measurement in zip(shells, measurements):
                if side == 'inner' and measurement['shell_voxels']/count > POLICY.maximum_inner_foreground_fraction:
                    continue
                if measurement['error_coverage'] >= POLICY.minimum_error_coverage:
                    selected |= errors & shell
                    entry.update(selected_width_mm=measurement['width_mm'],
                        selected_errors=measurement['covered_errors'], reason='active')
                    break
        report['sides'][side] = entry
    report.update(selected_errors=int(selected.sum()),
        total_error_coverage=int(selected.sum())/max(1,fp_count+fn_count))
    return selected, report


class ResidualBandLoss(torch.nn.Module):
    """BCE on selected mistakes, divided by fixed GT size—not by error count."""
    sink = None

    def __init__(self, foreground_class_ids=(1,), complement_class_ids=(0,), *, envelope=False):
        super().__init__()
        if tuple(foreground_class_ids)!=(1,) or tuple(complement_class_ids)!=(0,):
            raise ValueError('Conditional HFH BAND is binary only')
        self.envelope = envelope

    def forward(self, logits, labels):
        if tuple(logits.shape)!=(1,2,64,64,64) or tuple(labels.shape)!=(1,1,64,64,64):
            raise ValueError('Conditional BAND accounting requires the frozen HFH grid')
        values = labels.detach().cpu().numpy()[0,0]
        if not np.isin(values,[0,1]).all():
            raise ValueError('Nonbinary annotation')
        annotation = values.astype(np.uint8).tobytes()
        truth, roi, inner, outer = geometry(annotation)
        if self.envelope:
            support, report = inner[-1] | outer[-1], {'calibration_envelope':True}
        else:
            prediction = logits.detach().float().argmax(1)[0].cpu().numpy().astype(bool)
            support, report = selection(truth,prediction,roi,inner,outer)
            report['annotation_sha256']=hashlib.sha256(annotation).hexdigest()
            sink = type(self).sink
            if sink is not None:
                sink(report)
        with torch.autocast(logits.device.type, enabled=False):
            log_odds=logits[:,1].float()-logits[:,0].float()
            if support.any():
                mask=torch.as_tensor(support.copy(),device=logits.device)
                target=torch.as_tensor(truth.astype(np.float32),device=logits.device)
                loss=(F.binary_cross_entropy_with_logits(log_odds[0],target,reduction='none')*mask).sum()/int(truth.sum())
            else:
                loss=differentiable_zero(log_odds)
        return ConstraintResult(loss, torch.exp(-loss.detach()), loss.detach(), report)


def policy_dict():
    values=asdict(POLICY)
    # JSON round trips must preserve the exact frozen policy representation.
    values['spacing_mm']=list(values['spacing_mm'])
    values['widths_mm']=list(values['widths_mm'])
    return dict(version=VERSION, **values,
        normalization='Sum of BCE on selected errors divided by fixed GT foreground count',
        geometry='Anisotropic voxel-center EDT minus 0.5 mm; candidates in physical mm',
        scope='Training-case only; hard decisions detached; no held-out feedback',
        static_geometry='Label-only preprocessing, outside optimization FLOPs; cache by annotation bytes',
        extra_optimization_flops_per_active_update=EXTRA_UPDATE_FLOPS)
