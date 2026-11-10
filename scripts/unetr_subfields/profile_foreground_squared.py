"""Reproduce original Swin cost, then profile the foreground squared Dice variant."""
from pathlib import Path
import torch
import monai
from scripts.unetr_subfields import profile_baselines as previous,foreground_squared_dice
from scripts.unetr_subfields.contracts import ROOT
from scripts.cross_dataset_campaign.common import read,write,digest


def main():
    assert torch.__version__.startswith('2.2.2') and monai.__version__=='1.6.0'
    torch.set_num_threads(4);output=ROOT/'reports/mni_baseline_rescue_20261106/profiles'
    control=previous.measure('swinunetr',4)
    original=read(ROOT/'reports/mni_other_baselines_20261106/profiles/swinunetr.json')
    assert control['update_flops']==original['update_flops'] and control['parameters']==original['parameters']
    write(output/'swin_original_dice_control.json',dict(**control,status='exact_previous_recipe_match'))
    previous.training_loss=foreground_squared_dice.training_loss
    result=previous.measure('swinunetr',4)
    result.update(loss_recipe='foreground_squared_dice',loss_arguments=foreground_squared_dice.LOSS_ARGUMENTS,
        loss_adapter_sha256=digest(Path(foreground_squared_dice.__file__)),
        recipe=dict(alias='swinunetr',lr=.0003,loss='Foreground-only squared softmax Dice, no CE; smooth_nr0,smooth_dr1e-6; AdamW'))
    write(output/'swin_foreground_squared.json',result)
    print('Original',control['update_flops'],'foreground_squared',result['update_flops'],flush=True)


if __name__=='__main__':main()
