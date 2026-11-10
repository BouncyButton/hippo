"""Verify and collate all completed conditions at the 31,976.45 GFLOP cap."""
import json
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUDGET = 31976450667090
CAMPAIGNS = (
    ('four_models_six_optimization_31976_20261006', 685853, 'original'),
    ('nnunet_early_calibration_31976_20261006', 686327, 'early_calibration'),
    ('nnunet_delayed_sdf_31976_20261006', 686438, 'delayed_sdf_with_bands'),
    ('nnunet_delayed_sdf_only_31976_20261006', 686501, 'delayed_sdf_only'),
)
MODELS = ('nnunet', 'swinunetr', 'unetr_vit_b16', 'basic_unet16')
ARMS = ('dice', 'dice_bands', 'dice_sdf', 'dice_bands_sdf',
        'dice_bands_edge', 'dice_bands_edge_sdf')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    rows = []
    inputs = {}
    train_cases = validation_cases = dataset_sha256 = None
    for campaign, job, variant in CAMPAIGNS:
        folder = ROOT / 'reports' / campaign
        summary_path = folder / 'summary.json'
        summary = json.loads(summary_path.read_text())
        assert summary['status'] == 'complete' and not summary.get('failures')
        inputs[str(summary_path.relative_to(ROOT))] = digest(summary_path)
        metadata = folder / 'runs_metadata' if variant == 'original' else folder
        for original in summary['runs']:
            row = dict(original)
            run = metadata / row['model'] / row['objective'] / 'fold0/seed0/run'
            done = json.loads((run / 'completion.json').read_text())
            config = json.loads((run / 'config.json').read_text())
            val = json.loads((run / 'validation_latest.json').read_text())
            for name, expected in (
                ('completion.json', row['completion_sha256']),
                ('validation_latest.json', row['validation_latest_sha256']),
                ('runtime_recipe.json', row['runtime_recipe_sha256']),
                ('config.json', done['config_sha256']),
                ('calibration.json', done['calibration_sha256']),
                ('budget_ledger.json', done['budget_ledger_sha256']),
            ):
                path = run / name
                assert digest(path) == expected, str(path)
                inputs[str(path.relative_to(ROOT))] = expected
            if train_cases is None:
                train_cases = config['train_cases']
                validation_cases = config['validation_cases']
                dataset_sha256 = config['dataset_sha256']
            assert config['train_cases'] == train_cases
            assert config['validation_cases'] == validation_cases
            assert config['dataset_sha256'] == dataset_sha256
            assert len(train_cases) == 10 and len(validation_cases) == 52
            assert [case['case_name'] for case in val['cases']] == validation_cases
            assert row['fold'] == row['seed'] == 0
            assert done['status'] == 'complete' and done['budget_flops'] == BUDGET
            assert row['budget_scope'] == done['budget_scope'] == 'optimization_only'
            assert 0 < done['consumed_training_flops'] <= BUDGET
            assert row['optimization_gflops'] == done['consumed_training_flops'] / 1e9
            assert row['calibration_gflops'] == done['calibration_flops'] / 1e9
            assert row['total_gflops'] == done['consumed_total_flops'] / 1e9
            assert row['updates'] == done['updates'] == done['training_attempts_charged']
            assert row['epochs'] == row['updates'] / 10
            assert row['final_dice'] == val['summary']['macro_hard_dice']
            for field, key in (('anterior_dice', 'anterior_hard_dice'),
                               ('posterior_dice', 'posterior_hard_dice')):
                assert row[field] == val['summary'][key]
                mean = sum(case[key] for case in val['cases']) / 52
                assert abs(mean - row[field]) < 1e-12
            assert abs(row['final_dice'] - (row['anterior_dice'] + row['posterior_dice']) / 2) < 1e-12
            row.update(campaign=campaign, job_id=job, schedule_variant=variant,
                       schedule=config['schedule'],
                       calibration_update=config['schedule']['prefix_updates'],
                       constraint_ramp_updates=config['constraint_warmup_updates'],
                       native_objective=config.get('native_loss', 'Single-output MONAI Dice'),
                       metadata_directory=str(run.relative_to(ROOT)))
            if row['objective'] == 'dice':
                row['calibration_update'] = None
                row['constraint_first_update'] = None
            else:
                row['constraint_first_update'] = row['calibration_update'] + 1
            if 'sdf' in row['objective']:
                row['sdf_first_update'] = config['schedule'].get('sdf_prefix_updates', row['calibration_update']) + 1
                row['sdf_ramp_updates'] = config['schedule'].get('sdf_warmup_updates', config['constraint_warmup_updates'])
                row['constraint_first_update'] = min(row['constraint_first_update'], row['sdf_first_update']) if 'bands' in row['objective'] else row['sdf_first_update']
            rows.append(row)
    assert len(rows) == 31
    baselines = {r['model']: r for r in rows if r['objective'] == 'dice'}
    assert set(baselines) == set(MODELS)
    for row in rows:
        row.update(final_dice_percent=100 * row['final_dice'],
                   anterior_dice_percent=100 * row['anterior_dice'],
                   posterior_dice_percent=100 * row['posterior_dice'],
                   gain_vs_model_baseline_pp=100 * (row['final_dice'] - baselines[row['model']]['final_dice']))
    directory = ROOT / 'reports/capped_constraint_results_31976_20261006'
    directory.mkdir(exist_ok=True)
    result = dict(budget_flops=BUDGET, budget_gflops=BUDGET / 1e9,
                  budget_scope='optimization_only', calibration_in_cap=False,
                  fold=0, seed=0, training_volumes=10, validation_volumes=52,
                  dataset_sha256=dataset_sha256, train_cases=train_cases,
                  validation_cases=validation_cases,
                  endpoint='Final checkpoint macro foreground hard Dice; gains in percentage points versus each model baseline at the same ceiling.',
                  runs=rows, total_conditions=len(rows),
                  all_result_metadata_sha256_verified=True,
                  input_sha256=inputs, generator_sha256=digest(Path(__file__)))
    (directory / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    main_rows = [r for r in rows if r['schedule_variant'] == 'early_calibration'
                 or (r['schedule_variant'] == 'original' and (r['model'] != 'nnunet' or r['objective'] == 'dice'))]
    print('Main table: final Dice % (gain pp)')
    for arm in ARMS:
        cells = [arm]
        for model in MODELS:
            r = next(r for r in main_rows if r['model'] == model and r['objective'] == arm)
            cells.append(f"{r['final_dice_percent']:.2f} ({r['gain_vs_model_baseline_pp']:+.2f})")
        print(' | '.join(cells))
    print('Additional nnU-Net conditions:')
    for r in rows:
        if (r['model'] == 'nnunet' and r['objective'] != 'dice'
                and r['schedule_variant'] != 'early_calibration'):
            print(f"{r['schedule_variant']} | {r['objective']} | {r['updates']} | {r['final_dice_percent']:.2f} | {r['gain_vs_model_baseline_pp']:+.2f}")
    for model in MODELS:
        candidates = [r for r in rows if r['model'] == model]
        best = max(candidates, key=lambda r: r['final_dice'])
        print(f"Best observed {model}: {best['objective']} {best['schedule_variant']} {best['final_dice_percent']:.2f} ({best['gain_vs_model_baseline_pp']:+.2f})")
    print(json.dumps({'all_conditions_verified': len(rows), 'report': str(directory / 'summary.json')}))


if __name__ == '__main__':
    main()
