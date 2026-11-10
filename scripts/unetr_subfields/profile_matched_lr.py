"""Reprofile Swin Dice+CE; retain nnUNet's independently reproduced update cost."""
from pathlib import Path
import torch
import monai
from scripts.unetr_subfields import profile_baselines as previous, ce_loss
from scripts.unetr_subfields.contracts import ROOT
from scripts.cross_dataset_campaign.common import read,write,digest

NAME='mni_matched_lr_dice_ce_20261106'


def main():
    assert torch.__version__.startswith('2.2.2') and monai.__version__=='1.6.0'
    torch.set_num_threads(4);output=ROOT/'reports'/NAME/'profiles'
    for identity in ('nnunet','swinunetr'):
        control=previous.measure(identity,4)
        old=read(ROOT/'reports/mni_other_baselines_20261106/profiles'/(identity+'.json'))
        assert control['update_flops']==old['update_flops'] and control['parameters']==old['parameters']
        write(output/(identity+'_control.json'),dict(**control,status='exact_previous_recipe_match'))
        if identity=='swinunetr':
            previous.training_loss=ce_loss.training_loss
            result=previous.measure(identity,4)
            assert result['update_flops']>control['update_flops']
        else:result=control
        result['recipe']=dict(alias=identity,lr=.01 if identity=='nnunet' else .0001,
            loss='Native foreground Dice+CE with deep supervision; SGD/Nesterov momentum0.95' if identity=='nnunet' else 'Existing MONAI softmax Dice including background + mean cross-entropy, weights1:1; AdamW')
        result['loss_recipe']='dice_ce'
        result['matched_lr_profile_source_sha256']=digest(Path(__file__))
        result['loss_adapter_sha256']=digest(ROOT/'scripts/unetr_subfields/ce_loss.py') if identity=='swinunetr' else digest(ROOT/'scripts/nnunet_multi_objective.py')
        write(output/(identity+'.json'),result)
        print(identity,result['update_flops'],flush=True)

if __name__=='__main__':main()
