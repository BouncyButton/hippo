"""Foreground-only squared softmax Dice, without cross-entropy."""
from monai.losses import DiceLoss

LOSS_ARGUMENTS=dict(include_background=False,to_onehot_y=True,softmax=True,
                    squared_pred=True,smooth_nr=0.,smooth_dr=1e-6)


def training_loss(outputs,labels,arm,weights,scale,target_sdf=None,criterion=None,depth_weights=None,sdf_scale=0.):
    if arm!='dice' or weights or any(v is not None for v in (target_sdf,criterion,depth_weights)):
        raise ValueError('Only the foreground squared Dice-only baseline is supported')
    return DiceLoss(**LOSS_ARGUMENTS)(outputs.float(),labels)
