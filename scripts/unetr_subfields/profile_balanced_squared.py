"""Profile class-balanced Dice after reproducing the unweighted Dice control."""
from functools import partial
from pathlib import Path
import torch
import monai
from scripts.unetr_subfields import profile_baselines as previous,foreground_squared_dice,balanced_squared_dice
from scripts.unetr_subfields.contracts import ROOT
from scripts.cross_dataset_campaign.common import read,write,digest


def main():
    assert torch.__version__.startswith('2.2.2') and monai.__version__=='1.6.0'
    torch.set_num_threads(4);output=ROOT/'reports/mni_baseline_rescue_20261106/profiles'
    previous.training_loss=foreground_squared_dice.training_loss
    control=previous.measure('swinunetr',4);assert control['update_flops']==read(output/'swin_foreground_squared.json')['update_flops']
    weights=read(output.parent/'swin_training_class_weights.json')['folds']['0']['weights']
    previous.training_loss=partial(balanced_squared_dice.training_loss,class_weights=weights)
    original=previous.baseline_runtime.build_model
    previous.baseline_runtime.build_model=lambda *args,**kwargs:balanced_squared_dice.initialize_head(original(*args,**kwargs))
    result=previous.measure('swinunetr',4)
    result.update(loss_recipe='balanced_foreground_squared_dice',loss_arguments=foreground_squared_dice.LOSS_ARGUMENTS,
        profile_class_weights=weights,output_head_initialization='zeros (uniform initial softmax)',
        loss_adapter_sha256=digest(Path(balanced_squared_dice.__file__)),
        recipe=dict(alias='swinunetr',lr=.0003,loss='Foreground-only squared Dice, inverse-training-volume class weights normalized to mean1; noCE; zero-initialized outputhead; AdamW'))
    write(output/'swin_balanced_squared.json',result)
    print('Unweighted',control['update_flops'],'balanced',result['update_flops'],flush=True)


if __name__=='__main__':main()
