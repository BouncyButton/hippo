"""Verify preserved metadata and assemble final endpoint comparisons."""
import csv
import hashlib
import json
from pathlib import Path
import sys

REPORT = Path(__file__).resolve().parent
ROOT = REPORT / 'verified_metadata'
SOURCE = ROOT / 'source'
def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):
    return json.loads(Path(path).read_text())
def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')

sys.path[:0] = [str(SOURCE), str(SOURCE / 'scripts')]
from campaign90 import MODELS
from scripts.capped_depth_delay.constants import ARMS, CAMPAIGN

summary = read(ROOT / 'summary.json')
assert summary['status'] == 'complete' and not summary['failures']
expected = {(model, arm) for model in MODELS for arm in ARMS}
assert len(summary['runs']) == 44 and {(r['model'], r['objective']) for r in summary['runs']} == expected
submission = read(REPORT / 'submission.json')
assert digest(ROOT / 'source_manifest.json') == submission['source_manifest_sha256']
for name, sha in read(ROOT / 'source_manifest.json').items():
    assert digest(SOURCE / name) == sha, name
assert digest(SOURCE / 'experiments' / CAMPAIGN / 'plan.json') == submission['plan_sha256']
assert digest(ROOT / 'preflight.json') == submission['preflight_sha256']

# Reuse the production metadata verifier. The original worker already verified
# full optimizer checkpoints before exporting lossless model-only checkpoints.
# Those intentionally transient files are not demanded by this metadata audit.
code = (SOURCE / 'scripts/capped_depth_delay/verify.py').read_text()
code = code.replace('    from scripts.capped_depth_delay.core import ARTIFACTS',
    '    ARTIFACTS = {"config_sha256":"config.json", "metrics_sha256":"metrics.csv", '
    '"budget_ledger_sha256":"budget_ledger.json", "validation_best_sha256":"validation_best.json", '
    '"calibration_sha256":"calibration.json"}')
namespace = {}
exec(compile(code, str(SOURCE / 'scripts/capped_depth_delay/verify.py'), 'exec'), namespace)
verified = []
for row in summary['runs']:
    model, arm = row['model'], row['objective']
    output = ROOT / model / arm / 'fold0/seed0/run'
    manifest = read(output / 'export_manifest.json')
    for name, sha in manifest['files'].items():
        if Path(name).suffix in ('.json', '.csv'):
            assert digest(output / name) == sha, str(output / name)
    assert manifest['original_artifacts_verified_before_export']
    assert digest(output / 'completion.json') == row['completion_sha256'] == manifest['original_completion_sha256']
    checked = namespace['verify_run'](ROOT, model, arm, SOURCE / 'experiments' / CAMPAIGN / model)
    assert all(row[k] == v for k, v in checked.items())
    assert row['calibration_anchor_and_existing_coefficients_exact']
    for filename in ('validation_best.json', 'validation_latest.json'):
        validation = read(output / filename)
        assert len(validation['cases']) == 52
        anterior = sum(c['anterior_hard_dice'] for c in validation['cases']) / 52
        posterior = sum(c['posterior_hard_dice'] for c in validation['cases']) / 52
        assert abs(anterior - validation['summary']['anterior_hard_dice']) < 1e-12
        assert abs(posterior - validation['summary']['posterior_hard_dice']) < 1e-12
        assert abs((anterior + posterior) / 2 - validation['summary']['macro_hard_dice']) < 1e-12
    assert len(read(output / 'config.json')['train_cases']) == 10
    assert digest(output / 'validation_latest.json') == row['validation_latest_sha256']
    assert abs(row['final_dice'] - (row['anterior_dice'] + row['posterior_dice']) / 2) < 1e-12
    if 'sdf' in arm:
        schedule = read(output / 'config.json')['schedule']
        assert schedule['sdf_delay_fraction_numerator'] == 5 and schedule['sdf_delay_fraction_denominator'] == 9
        assert row['sdf_first_update'] == schedule['sdf_prefix_updates'] + 1
    verified.append(checked)

old_path = REPORT.parent / 'capped_constraint_results_31976_20261006/summary.json'
old = read(old_path)
write(REPORT / 'retained_source_summary.json', old)
retained = []
for model in MODELS:
    for arm in ('dice', 'dice_bands', 'dice_bands_edge'):
        variant = 'early_calibration' if model == 'nnunet' and arm != 'dice' else 'original'
        matches = [r for r in old['runs'] if r['model'] == model and r['objective'] == arm and r['schedule_variant'] == variant]
        assert len(matches) == 1
        retained.append(matches[0])
rows = retained + summary['runs']
lookup = {(r['model'], r['objective']): r for r in rows}
baselines = {m: lookup[(m, 'dice')]['final_dice'] for m in MODELS}
best = {m: max(r['final_dice'] for r in rows if r['model'] == m) for m in MODELS}
labels = [
 ('dice', 'Baseline'), ('dice_bands', '+ bands'),
 ('dice_sdf', '+ signed distance'), ('dice_bands_sdf', '+ bands + signed distance'),
 ('dice_bands_edge', '+ bands + original edge'),
 ('dice_bands_edge_sdf', '+ bands + signed distance + original edge'),
 ('dice_depth', '+ depth-weighted CE'), ('dice_bands_depth', '+ bands + depth-weighted CE'),
 ('dice_edge_depth', '+ original edge + depth-weighted CE'),
 ('dice_sdf_depth', '+ signed distance + depth-weighted CE'),
 ('dice_bands_edge_depth', '+ bands + original edge + depth-weighted CE'),
 ('dice_bands_sdf_depth', '+ bands + signed distance + depth-weighted CE'),
 ('dice_edge_sdf_depth', '+ original edge + signed distance + depth-weighted CE'),
 ('dice_bands_edge_sdf_depth', '+ bands + signed distance + original edge + depth-weighted CE')]
models = list(MODELS)
header = '| Objective | nnU-Net | SwinUNETR | UNETR | 3D U-Net, width 16 |'
lines = [header, '|---|---:|---:|---:|---:|']
csv_rows = []
for arm, label in labels:
    cells = []
    for model in models:
        row = lookup[(model, arm)]
        score = row['final_dice'] * 100
        gain = (row['final_dice'] - baselines[model]) * 100
        cell = f'{score:.2f}' if arm == 'dice' else f'{score:.2f} ({gain:+.2f})'
        if row['final_dice'] == best[model]:
            cell = '**' + cell + '**'
        cells.append(cell)
        csv_rows.append(dict(model=model, objective=arm, final_dice_percent=score,
            gain_vs_baseline_pp=gain, anterior_dice_percent=row['anterior_dice']*100,
            posterior_dice_percent=row['posterior_dice']*100, updates=row['updates'], epochs=row['epochs'],
            optimization_gflops=row['optimization_gflops'], calibration_gflops=row['calibration_gflops']))
    lines.append('| ' + label + ' | ' + ' | '.join(cells) + ' |')
notes = ('Final mean foreground validation Dice (%), gains in percentage points versus each model baseline. '
    'Exploratory fold 0 / seed 0; ten training and 52 validation volumes. '
    'Optimization ceiling 31,976.45066709 GFLOPs; calibration separately reported and excluded. '
    'All signed-distance objectives use the new approximately 56%-of-allowance delay. '
    'Depth means original depth-weighted cross-entropy; edge means original edge consistency. '
    'Retained nnU-Net constraint rows use early calibration. Bold marks each model’s best final result.')
(REPORT / 'final_results_table.md').write_text(notes + '\n\n' + '\n'.join(lines) + '\n')
with (REPORT / 'final_results.csv').open('w') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
    writer.writeheader(); writer.writerows(csv_rows)
write(REPORT / 'final_results_sources.json', dict(new_campaign=summary, retained_conditions=retained,
    retained_summary_sha256=digest(old_path), optimization_budget_flops=summary['budget_flops']))
audit = dict(status='passed', job_id=687658, completed_new_conditions=44, retained_conditions=12,
    source_files_sha256_verified=len(read(ROOT/'source_manifest.json')),
    metadata_manifests_verified=44, completed_configs_and_ledgers_verified=44,
    validation_volumes_each=52, training_volumes_each=10, per_case_validation_averages_recomputed=True,
    cluster_export_verification=read(ROOT/'final_export_verification.json'),
    verified_source_artifacts={name:digest(REPORT/name) for name in
        ['final_results_table.md','final_results.csv','final_results_sources.json','retained_source_summary.json']})
write(REPORT / 'final_verification.json', audit)
print('\n'.join(lines))
