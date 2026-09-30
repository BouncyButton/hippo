import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_edge_lowdata import data_arguments, reference_directories


@pytest.mark.parametrize('fold', [0, 1, 2])
def test_requested_fold_reaches_data_loader(fold):
    config = {'pkl': '/data/full.pkl', 'splits_json': f'/data/fold{fold}.json',
              'run': {'fold': fold, 'spatial_size': [64, 64, 64], 'resize': False}}
    args = data_arguments(config)
    assert args.fold == fold
    assert args.splits_json == Path(f'/data/fold{fold}.json')
    assert args.plain_tensors and not args.resize


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_fold_and_seed_specific_references(seed):
    base = Path('/cluster')
    fold0 = reference_directories(base, 0, seed)
    assert fold0['bands'] == base/f'low_data_noaug_seed{seed}_20260917_01/runs/bands_5pct_seed{seed}_noaug'
    for fold in (1, 2):
        refs = reference_directories(base, fold, seed)
        assert refs['dice'] == base/f'low_data_noaug_fold{fold}_20260917_01/runs/baseline_seed{seed}'
        assert refs['bands'] == base/f'low_data_noaug_fold{fold}_20260917_01/runs/bands_seed{seed}'


def test_unsupported_fold_fails_instead_of_reusing_fold0():
    with pytest.raises(ValueError):
        reference_directories(Path('/cluster'), 4, 0)
