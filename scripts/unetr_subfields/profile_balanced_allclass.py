"""Profile weighted squared Dice including explicit background supervision."""
from functools import partial
from pathlib import Path
import torch
import monai
from scripts.unetr_subfields import profile_baselines as previous,balanced_squared_dice
from scripts.unetr_subfields.contracts import ROOT
from scripts.cross_dataset_campaign.common import read,write,digest


def main():
    assert torch.__version__.startswith('2.2.2') and monai.__version__=='1.6.0'
    torch.set_num_threads(4);output=ROOT/'reports/mni_baseline_rescue_20261106/profiles'
    foreground=read(output.parent/'swin_training_class_weights.json')['folds']['0']['weights']
    original=previous.baseline_runtime.build_model
    previous.baseline_runtime.build_model=lambda *a,**kw:balanced_squared_dice.initialize_head(original(*a,**kw))
    previous.training_loss=partial(balanced_squared_dice.training_loss,class_weights=foreground)
    control=previous.measure('swinunetr',4);assert control['update_flops']==read(output/'swin_balanced_squared.json')['update_flops']
    weights=[1.,*foreground];previous.training_loss=partial(balanced_squared_dice.training_loss,class_weights=weights,include_background=True)
    result=previous.measure('swinunetr',4)
    result.update(loss_recipe='balanced_allclass_squared_dice',loss_arguments={**balanced_squared_dice.LOSS_ARGUMENTS,'include_background':True},
        profile_class_weights=weights,output_head_initialization='zeros (uniform initial softmax)',
        loss_adapter_sha256=digest(Path(balanced_squared_dice.__file__)),
        recipe=dict(alias='swinunetr',lr=.0003,loss='All-class squared Dice, backgroundweight1 and inverse-training-volume foregroundweights(mean1); noCE; zero outputhead; AdamW'))
    write(output/'swin_balanced_allclass.json',result)
    print('Foreground',control['update_flops'],'allclass',result['update_flops'],flush=True)


if __name__=='__main__':main()
