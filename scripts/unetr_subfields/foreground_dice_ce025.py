"""Foreground squared Dice plus one-quarter dense cross-entropy."""
from monai.losses import DiceCELoss
from scripts.unetr_subfields.foreground_squared_dice import LOSS_ARGUMENTS as DICE_ARGUMENTS

LOSS_ARGUMENTS=dict(DICE_ARGUMENTS,lambda_dice=1.,lambda_ce=.25)


def training_loss(outputs,labels,arm,weights,scale,target_sdf=None,criterion=None,depth_weights=None,sdf_scale=0.):
    if arm!='dice' or weights or any(v is not None for v in (target_sdf,criterion,depth_weights)):
        raise ValueError('Only the Dice plus quarter-CE base is supported')
    return DiceCELoss(**LOSS_ARGUMENTS)(outputs.float(),labels)
