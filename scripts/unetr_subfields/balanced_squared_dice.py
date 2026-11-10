"""Training-frequency-balanced foreground Dice and a uniform output head."""
from monai.losses import DiceLoss
import torch
from scripts.unetr_subfields.foreground_squared_dice import LOSS_ARGUMENTS


def initialize_head(model):
    head=model.out.conv.conv
    assert isinstance(head,torch.nn.Conv3d) and head.out_channels==4 and head.kernel_size==(1,1,1)
    torch.nn.init.zeros_(head.weight)
    if head.bias is not None:torch.nn.init.zeros_(head.bias)
    return model


def training_loss(outputs,labels,arm,weights,scale,target_sdf=None,criterion=None,depth_weights=None,sdf_scale=0.,*,class_weights,include_background=False):
    if arm!='dice' or weights or any(v is not None for v in (target_sdf,criterion,depth_weights)):
        raise ValueError('Only balanced foreground squared Dice is supported')
    if len(class_weights)!=(4 if include_background else 3) or any(v<=0 for v in class_weights):raise ValueError('Unexpected class weights')
    arguments={**LOSS_ARGUMENTS,'include_background':include_background}
    return DiceLoss(**arguments,weight=class_weights)(outputs.float(),labels)
