"""Historical softmax Dice plus unweighted mean multiclass CE, with equal weights."""
import torch.nn.functional as F
from scripts.compute_constraint_objectives import training_loss as dice_loss


def training_loss(logits, labels, arm, weights, scale, target_sdf=None,
                  criterion=None, depth_weights=None, sdf_scale=0.):
    if arm not in ('dice', 'dice_ce') or weights or criterion is not None or target_sdf is not None or depth_weights is not None:
        raise ValueError('This adapter supports only unweighted UNETR Dice+CE')
    return dice_loss(logits, labels, 'dice', {}, scale) + F.cross_entropy(
        logits.float(), labels[:, 0].long(), reduction='mean')
