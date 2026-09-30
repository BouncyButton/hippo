"""Unbiased adjacent-slice estimators of the original supervised 3-D losses."""
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from thesis.new_constraints.bands.outer_boundary import build_boundary_bands
from thesis.new_constraints.edge_consistency import signed_face_error


@dataclass
class VolumeSupport:
    foreground: torch.Tensor  # [Z,Y,X], native canonical grid
    inner: torch.Tensor
    outer: torch.Tensor
    inner_count: int
    outer_count: int
    face_count: int

    @classmethod
    def build(cls, foreground):
        foreground = foreground.bool()
        if foreground.ndim != 3:
            raise ValueError('Expected native [Z,Y,X] foreground.')
        inner, outer = build_boundary_bands(foreground[None, None], steps=2)
        _, count = signed_face_error(torch.zeros_like(foreground)[None, None].float(),
                                     foreground[None, None], inner | outer)
        return cls(foreground, inner[0, 0], outer[0, 0],
                   int(inner.sum()), int(outer.sum()), int(count.item()))


def semantic_log_fields(output, shape):
    """Stable foreground/complement logs for the winning instance at each pixel."""
    masks = output['pred_masks']
    if masks.shape[0] != 1:
        raise ValueError('The adjacent-slice estimator requires batch size one.')
    score_logits = output['pred_logits'][0].float().reshape(-1)
    presence_logits = output['presence_logit_dec'][0].float().reshape(-1)
    scores, presence = F.logsigmoid(score_logits), F.logsigmoid(presence_logits)
    if presence.numel() != 1 or scores.numel() != masks.shape[1]:
        raise ValueError('Unexpected SAM3 confidence tensor shapes.')
    mask_logits = F.interpolate(masks[0, :, None].float(), size=shape,
                                mode='bilinear', align_corners=False)[:, 0]
    log_masks = F.logsigmoid(mask_logits)
    log_p, winner = (log_masks + (scores + presence)[:, None, None]).max(0)
    winning_mask = mask_logits.gather(0, winner[None])[0]
    winning_score = score_logits[winner]
    lm, ls = F.logsigmoid(winning_mask), F.logsigmoid(winning_score)
    # 1-m*s*r = (1-m) + m*(1-s) + m*s*(1-r). This avoids cancellation
    # and overflowing complement gradients even for logits of +/-100.
    log_q = torch.logsumexp(torch.stack((F.logsigmoid(-winning_mask),
                              lm+F.logsigmoid(-winning_score),
                              lm+ls+F.logsigmoid(-presence_logits))), dim=0)
    return log_p, log_q


def semantic_log_probability(output, shape):
    return semantic_log_fields(output, shape)[0]


def semantic_probability(output, shape):
    return semantic_log_probability(output, shape).exp()


def sampled_losses(probability, next_probability, support, anchor, *, log_probability=None,
                   log_complement=None):
    """Uniform-anchor expectation equals full-volume bands and signed-face loss.

    The second native training slice wraps cyclically to preserve uniform
    exposure. Anatomical faces never wrap. Full-volume denominators are fixed
    from training labels, avoiding slice-size-dependent reweighting.
    """
    depth = support.foreground.shape[0]
    if not 0 <= anchor < depth or probability.shape != support.foreground.shape[1:]:
        raise ValueError('Invalid anchor or native probability shape.')
    if probability.shape != next_probability.shape:
        raise ValueError('Adjacent probability grids differ.')
    device = probability.device
    truth = support.foreground[anchor].to(device)
    inner, outer = support.inner[anchor].to(device), support.outer[anchor].to(device)
    zero = (probability.sum() + next_probability.sum()) * 0
    band = zero
    if support.inner_count and support.outer_count:
        log_p = probability.log() if log_probability is None else log_probability
        log_q = torch.log1p(-probability) if log_complement is None else log_complement
        bce = F.binary_cross_entropy_with_logits(log_p-log_q, truth.float(), reduction='none')
        band = .5 * depth * ((bce * inner).sum()/(support.inner_count+1e-6)
                             + (bce * outer).sum()/(support.outer_count+1e-6))
    residual = probability - truth.to(probability.dtype)
    active = inner | outer
    total = zero
    for axis in (0, 1):
        left, right = [slice(None)]*2, [slice(None)]*2
        left[axis], right[axis] = slice(None, -1), slice(1, None)
        left, right = tuple(left), tuple(right)
        valid = active[left] & active[right]
        total = total + ((residual[left]-residual[right]).square()*valid).sum()
    if anchor < depth-1:
        next_truth = support.foreground[anchor+1].to(device)
        next_active = (support.inner[anchor+1] | support.outer[anchor+1]).to(device)
        next_residual = next_probability - next_truth.to(next_probability.dtype)
        total = total + ((residual-next_residual).square()*(active & next_active)).sum()
    edge = depth * total / max(support.face_count, 1)
    return band, edge
