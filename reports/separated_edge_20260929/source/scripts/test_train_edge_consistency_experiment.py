import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_edge_consistency_experiment import experiment_run


def test_preserves_reference_settings_and_binds_calibration(tmp_path):
    path = tmp_path / 'probe.json'
    path.write_text('{}')
    reference = {'run': {'constraint_set': 'bands', 'constraint_config': {'bands_weight': .012},
                         'optimizer_mode': 'adamw_0.01', 'seed': 0, 'calibration_diagnostics': True}}
    original = copy.deepcopy(reference)
    probe = {'gate': {'passed': True}, 'calibration': {'band_weight': .012, 'edge_weight': .03}}
    run = experiment_run(reference, probe, path, 'source', 'runtime', 'execution')
    assert reference == original
    assert run['constraint_config'] == reference['run']['constraint_config']
    assert run['optimizer_mode'] == 'adamw_0.01' and run['seed'] == 0
    assert run['edge_weight'] == .03 and run['constraint_set'] == 'bands_edge'
    assert len(run['edge_calibration_sha256']) == 64
    for mutation in ({'gate': {'passed': False}},
                     {'calibration': {'band_weight': .015, 'edge_weight': .03}},
                     {'calibration': {'band_weight': .012, 'edge_weight': float('nan')}}):
        with pytest.raises(ValueError):
            experiment_run(reference, probe | mutation, path, 's', 'r', 'e')
