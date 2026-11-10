"""Reproduce the original Swin cost before auditing all-class log Dice."""
from pathlib import Path
import torch
import monai
from scripts.unetr_subfields import profile_baselines as previous, log_squared_dice
from scripts.unetr_subfields.contracts import ROOT
from scripts.cross_dataset_campaign.common import read, write, digest


def main():
    assert torch.__version__.startswith('2.2.2') and monai.__version__ == '1.6.0'
    torch.set_num_threads(4)
    output = ROOT / 'reports/mni_baseline_rescue_20261106/profiles'
    control = previous.measure('swinunetr', 4)
    original = read(ROOT / 'reports/mni_other_baselines_20261106/profiles/swinunetr.json')
    assert control['update_flops'] == original['update_flops']
    inherited_supplement = previous.supplement
    def with_log(func, args, kwargs, result):
        if str(func) == 'aten.log.default':
            return result.numel(), 'scalar_log_estimate'
        return inherited_supplement(func, args, kwargs, result)
    previous.supplement = with_log
    previous.training_loss = log_squared_dice.training_loss
    result = previous.measure('swinunetr', 4)
    result.update(loss_recipe='log_squared_dice', loss_arguments=log_squared_dice.LOSS_ARGUMENTS,
                  loss_adapter_sha256=digest(Path(log_squared_dice.__file__)),
                  recipe=dict(alias='swinunetr', lr=.0003,
                              loss='Mean negative log squared softmax Dice over all4 classes; no CE; AdamW'))
    result['accounting'] += ' Scalar log counts as one operation, like scalar exp in the historical convention.'
    write(output / 'swin_log_squared.json', result)
    write(output / 'swin_log_original_control.json', dict(**control, status='exact_previous_recipe_match'))
    print('Original', control['update_flops'], 'log_squared', result['update_flops'], flush=True)


if __name__ == '__main__':
    main()
