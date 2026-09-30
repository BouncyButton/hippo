import json
from pathlib import Path

import pytest

from scripts.run_degree_bands_ab import quota_free_bytes, training_arguments


def test_quota_parser_uses_user_limit_not_group_usage():
    text = '1009 1009 group pool_home 9.92TiB/∞ 38.76M/∞\n3160552 1395 user pool_home 90.67GiB/93.13GiB 169.14k/∞'
    assert quota_free_bytes(text) / 1024**3 == pytest.approx(2.46, abs=1e-8)
    with pytest.raises(RuntimeError):
        quota_free_bytes('quota service unavailable')


@pytest.mark.parametrize('normalization', ['inner', 'surface'])
def test_job_arguments_train_only_degree_bands_with_reference_stopping(tmp_path, normalization):
    reference = {'pkl': '/dataset.pkl', 'splits_json': '/splits.json', 'run': {
        'dataset': 'MSD', 'fold': 0, 'seed': 0, 'epochs': 50, 'batch_size': 1,
        'optimizer_mode': 'adamw_0.01', 'learning_rate': 1e-4, 'weight_decay': 1e-5,
        'step_size': 20, 'adamw_gamma': .5, 'training_augmentation': 'mild_v1',
        'drop_rate': 0, 'constraint_warmup_epochs': 5, 'early_stopping_patience': 8,
        'early_stopping_min_epochs': 25, 'early_stopping_min_delta': .0005,
        'spatial_size': [64, 64, 64], 'amp': True, 'activation_checkpointing': False,
        'plain_tensors': True, 'resize': False}}
    report = tmp_path / 'calibration.json'
    report.write_text(json.dumps({'recommended_bands_weight': .012}))
    arguments = training_arguments(reference, tmp_path / 'run', report, normalization, 2)
    expected = {'--constraint-set': 'bands', '--bands-degree-alpha': '2',
                '--bands-degree-normalization': normalization, '--early-stopping-patience': '8',
                '--early-stopping-min-epochs': '25', '--epochs': '50', '--bands-weight': '0.012'}
    for flag, value in expected.items():
        assert arguments[arguments.index(flag) + 1] == value
    assert '--init-checkpoint' not in arguments and '--resume' not in arguments
