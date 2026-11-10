"""Verify the completed capped recipe, metrics, calibration and compute ledger."""
import csv
import json
import math
from pathlib import Path
from scripts.campaign90 import BUDGET, MODELS, digest
from scripts.capped_depth_delay.schedule import training_flops, validation_updates
from .objectives import ARM
from .core import ARTIFACTS


def read(path):
    return json.loads(Path(path).read_text())


def verify(output, directory, model, fold, seed):
    protocol = read(directory/'protocol.json')
    config, done, ledger, cal = [read(output/name) for name in
        ('config.json', 'completion.json', 'budget_ledger.json', 'calibration.json')]
    alias = MODELS[model]['alias']
    schedule = protocol['models'][alias]['objectives'][ARM]
    split = read(directory/'splits_10patients.json')[fold]
    assert done['status'] == 'complete' and not config['smoke_test']
    assert (done['model'], done['objective'], done['fold'], done['seed']) == (alias, ARM, fold, seed)
    assert config['model_identity'] == model and config['schedule'] == schedule
    assert config['protocol_sha256'] == digest(directory/'protocol.json')
    assert config['dataset_sha256'] == protocol['pkl_sha256']
    assert config['budget_scope'] == done['budget_scope'] == 'optimization_only'
    assert config['budget_flops'] == BUDGET and config['lr_horizon_updates'] == MODELS[model]['lr_horizon']
    for key, name in ARTIFACTS.items():
        assert digest(output/name) == done[key], name
    n = schedule['max_steps']
    used = training_flops(n, schedule)
    assert used <= BUDGET < training_flops(n+1, schedule)
    assert done['updates'] == done['training_attempts_charged'] == ledger['charged_attempts'] == n
    assert done['discarded_attempts'] == 0
    assert ledger['training_flops'] == done['consumed_training_flops'] == used
    assert ledger['calibration_attempts'] == done['calibration_attempts'] == 1
    assert ledger['calibration_flops'] == done['calibration_flops'] == schedule['calibration_flops']
    assert done['consumed_total_flops'] == used + schedule['calibration_flops']
    assert done['unused_flops'] == BUDGET-used
    assert ledger['config_sha256'] == done['config_sha256'] == cal['config_sha256']
    assert ledger['calibration_sha256'] == done['calibration_sha256']
    assert cal['anchor_training_attempts'] == schedule['prefix_updates']
    assert [r['case_name'] for r in cal['cases']] == split['train']
    assert config['train_cases'] == split['train'] and config['validation_cases'] == split['val']
    assert set(cal['weights']) == {'bands','pair','depth'}
    assert all(math.isfinite(v) and v > 0 for v in cal['weights'].values())
    from .objectives import select_weights
    expected, _ = select_weights(cal['cases'], ARM)
    assert expected == cal['weights']
    rows = list(csv.DictReader((output/'metrics.csv').open()))
    assert [int(r['charged_attempts']) for r in rows] == validation_updates(protocol, ARM)
    for row in rows:
        step = int(row['charged_attempts'])
        assert int(row['step']) == step and int(row['estimated_training_flops']) == training_flops(step, schedule)
        assert math.isfinite(float(row['mean_train_loss']))
    best, final = [read(output/name) for name in ('validation_best.json','validation_latest.json')]
    for result in (best, final):
        assert [r['case_name'] for r in result['cases']] == split['val']
        score = sum((r['anterior_hard_dice']+r['posterior_hard_dice'])/2 for r in result['cases'])/52
        assert abs(score-result['summary']['macro_hard_dice']) < 1e-12
    score = best['summary']['macro_hard_dice']
    assert score == done['best_validation_hard_dice'] == max(float(r['macro_hard_dice']) for r in rows)
    assert final['summary']['macro_hard_dice'] == float(rows[-1]['macro_hard_dice'])
    return dict(model=model, objective=ARM, fold=fold, seed=seed, updates=n,
        epochs=n/10, best_dice=score, final_dice=final['summary']['macro_hard_dice'],
        anterior_dice=final['summary']['anterior_hard_dice'], posterior_dice=final['summary']['posterior_hard_dice'],
        optimization_gflops=used/1e9, calibration_gflops=schedule['calibration_flops']/1e9,
        unused_optimization_gflops=(BUDGET-used)/1e9, weights=cal['weights'])
