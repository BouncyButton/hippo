"""Measure the full Geo rescue recipe at the medical input shapes."""
from pathlib import Path
import torch
import monai
from scripts.unetr_subfields import profile, geo_recipe
from scripts.unetr_subfields.profile_dice_ce import supplement as ce_supplement
from scripts.unetr_subfields.contracts import ROOT, DATASETS
from scripts.cross_dataset_campaign.common import write, digest

NAME = 'unetr_subfields_geo_recipe_seed_search_20261106'


def supplement(func, args, kwargs, result):
    value, category = ce_supplement(func, args, kwargs, result)
    if category != 'no_supplement': return value, category
    name = str(func).split('.')[1] if str(func).startswith('aten.') else ''
    if name in ('native_dropout', 'native_dropout_backward'):
        return 2 * args[0].numel(), 'dropout_mask_and_scale_estimate'
    return value, category


def main():
    assert monai.__version__ == '1.6.0' and torch.__version__.startswith('2.2.2')
    torch.set_num_threads(4)
    output = ROOT / 'reports' / NAME / 'profiles'
    control = profile.measure(3, (64, 64, 64))
    assert control['update_flops'] == 130061273028
    write(output / 'historical_control.json', dict(**control, status='exact_match'))
    profile.build_model, profile.model_kwargs = geo_recipe.build_model, geo_recipe.model_kwargs
    profile.training_loss, profile.intermediate_supplement = geo_recipe.training_loss, supplement
    for dataset, spec in DATASETS.items():
        result = profile.measure(len(spec['classes']), spec['spatial_size'])
        result.update(dataset=dataset, loss_arguments=geo_recipe.LOSS_ARGUMENTS, dropout_rate=.1,
            runtime=dict(torch=torch.__version__, monai=monai.__version__, device='meta'),
            source_sha256={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),
                ROOT / 'scripts/unetr_subfields/geo_recipe.py', ROOT / 'scripts/unetr_subfields/profile_dice_ce.py',
                ROOT / 'scripts/unetr_subfields/profile.py', ROOT / 'scripts/unetr_subfields/core.py']},
            historical_control_sha256=digest(output / 'historical_control.json'),
            accounting='Historical MAC=2/scalar convention with CE estimates and dropout arithmetic; RNG, paired voxel permutation, validation and I/O excluded. Optimization only.')
        write(output / f'{dataset}.json', result)
        print(dataset, result['update_flops'], flush=True)


if __name__ == '__main__': main()
