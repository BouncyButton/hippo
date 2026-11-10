"""Mean negative log squared Dice over all classes, including background."""
import torch

LOSS_ARGUMENTS = dict(include_background=True, softmax=True, squared_pred=True,
                      smooth_nr=1e-6, smooth_dr=1e-6, logarithmic=True)


def training_loss(outputs, labels, arm, weights, scale, target_sdf=None,
                  criterion=None, depth_weights=None, sdf_scale=0.):
    if arm != 'dice' or weights or any(v is not None for v in (target_sdf, criterion, depth_weights)):
        raise ValueError('Only the logarithmic Dice-only baseline is supported')
    probabilities = outputs.float().softmax(dim=1)
    target = torch.zeros_like(probabilities).scatter_(1, labels.long(), 1.)
    axes = tuple(range(2, outputs.ndim))
    intersection = (probabilities * target).sum(dim=axes)
    denominator = (probabilities.square() + target.square()).sum(dim=axes)
    # Difference of logs avoids subtracting a near-one Dice loss to recover a tiny score.
    return ((denominator + 1e-6).log() - (2 * intersection + 1e-6).log()).mean()
