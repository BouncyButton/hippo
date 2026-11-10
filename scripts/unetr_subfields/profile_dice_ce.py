"""Audit Dice+CE using historical arithmetic and explicit fused CE supplements."""
from pathlib import Path
import torch
import monai
from scripts.unetr_subfields import profile
from scripts.unetr_subfields.ce_loss import training_loss
from scripts.unetr_subfields.contracts import ROOT, DATASETS
from scripts.cross_dataset_campaign.common import write, digest, read

NAME = 'unetr_subfields_dice_ce_seed_search_20261106'
BASE_SUPPLEMENT = profile.intermediate_supplement


def supplement(func, args, kwargs, result):
    value, category = BASE_SUPPLEMENT(func, args, kwargs, result)
    if category != 'no_supplement':
        return value, category
    name = str(func).split('.')[1] if str(func).startswith('aten.') else ''
    if name == '_log_softmax':
        return 4 * args[0].numel(), 'ce_log_softmax_estimate'
    if name == '_log_softmax_backward_data':
        n = args[0].numel()
        return 4 * n - n // args[1].shape[args[2]], 'ce_log_softmax_estimate'
    if name in ('nll_loss_forward', 'nll_loss2d_forward'):
        assert args[2] is None and args[3] == 1
        return 2 * args[1].numel(), 'ce_nll_mean_estimate'
    if name in ('nll_loss_backward', 'nll_loss2d_backward'):
        assert args[3] is None and args[4] == 1
        return 2 * args[2].numel(), 'ce_nll_mean_estimate'
    return value, category


def main():
    assert monai.__version__ == '1.6.0' and torch.__version__.startswith('2.2.2')
    torch.set_num_threads(4)
    output = ROOT / 'reports' / NAME / 'profiles'
    control = profile.measure(3, (64, 64, 64))
    assert control['update_flops'] == 130061273028
    write(output / 'historical_control.json', dict(**control, status='exact_match'))
    profile.training_loss = training_loss
    profile.intermediate_supplement = supplement
    for dataset, spec in DATASETS.items():
        result = profile.measure(len(spec['classes']), spec['spatial_size'])
        original = read(ROOT / 'reports/unetr_subfields_cap_20261106/profiles' / f'{dataset}.json')
        assert result['update_flops'] > original['update_flops']
        result.update(dataset=dataset, objective='dice_ce', lambda_dice=1., lambda_ce=1.,
            original_dice_update_flops=original['update_flops'],
            runtime=dict(torch=torch.__version__, monai=monai.__version__, device='meta'),
            historical_control_sha256=digest(output / 'historical_control.json'),
            source_sha256={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),
                ROOT / 'scripts/unetr_subfields/ce_loss.py', ROOT / 'scripts/unetr_subfields/profile.py',
                ROOT / 'scripts/unetr_subfields/core.py']},
            accounting='Historical MAC=2 and scalar estimates; fused log-softmax forward=4N, backward=4N-G; unweighted mean NLL forward/backward=2V each. Indexing and memory operations excluded. Optimization only.')
        write(output / f'{dataset}.json', result)
        print(dataset, result['update_flops'], 'additional', result['update_flops'] - original['update_flops'], flush=True)


if __name__ == '__main__':
    main()
