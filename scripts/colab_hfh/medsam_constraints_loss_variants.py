"""Explicit BAND side-weighting ablation; canonical 50/50 remains unchanged."""
from functools import wraps


def side_weighted(reference, inner_weight):
    if not 0 < inner_weight < 1:
        raise ValueError('Both supervised BAND sides must have positive weight')
    if inner_weight == .5:
        return reference

    @wraps(reference)
    def loss(probability, next_probability, support, anchor, *, log_probability=None, log_complement=None):
        import torch
        import torch.nn.functional as F
        original_band, edge = reference(probability, next_probability, support, anchor,
            log_probability=log_probability, log_complement=log_complement)
        if not support.inner_count or not support.outer_count:
            return original_band, edge
        truth = support.foreground[anchor].to(probability.device)
        inner = support.inner[anchor].to(probability.device)
        outer = support.outer[anchor].to(probability.device)
        lp = probability.log() if log_probability is None else log_probability
        lq = torch.log1p(-probability) if log_complement is None else log_complement
        bce = F.binary_cross_entropy_with_logits(lp-lq, truth.float(), reduction='none')
        depth = support.foreground.shape[0]
        band = depth*(inner_weight*(bce*inner).sum()/(support.inner_count+1e-6)
                      +(1-inner_weight)*(bce*outer).sum()/(support.outer_count+1e-6))
        return band, edge
    return loss


def install(original, protocol):
    weight = protocol['constraint_recipe'].get('inner_weight', .5)
    original.sampled_losses = side_weighted(original.sampled_losses, weight)
