"""Generate an unsubmitted experiment matrix and a timing-based budget.

This is a planning tool, not a training launcher. No cluster/network access.
Replication split and subject IDs deliberately remain unresolved.
"""
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parent / 'low_data_independent_audit_20260919/AUDIT.json'
METHODS = ['dice', 'bands', 'global_fg_bce', 'dice_ce_selected']
audit = json.loads(AUDIT.read_text())
observations = []
for source in audit['exploratory_timings']:
    groups = []
    for line in Path(source['log']).read_text().splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and 'timing_epoch' in row:
            if row['timing_epoch'] == 1:
                groups.append([])
            groups[-1].append(row)
    assert len(groups) == 3 and len(groups[1]) == 5
    for arm, group in [('dice', groups[0]), ('bands', groups[2])]:
        observations.append({
            'source_log': source['log'], 'arm': arm,
            'seconds_per_update': statistics.median(r['training_seconds'] for r in group[1:]) / 10,
            'seconds_per_52_case_validation': statistics.median(r['validation_seconds'] for r in group[1:]),
            'seconds_per_checkpoint_event': statistics.median(r['checkpoint_seconds'] for r in group[1:]),
        })

matrix = []


def add(stage, split, subset, seed, cases, method, steps, lr_step):
    key = f'{stage}_{split}_q{subset}_s{seed}_n{cases}_{method}_u{steps}'
    matrix.append({
        'run_id': key, 'stage': stage, 'split_role': split, 'subset_seed': subset,
        'model_seed': seed, 'training_case_count': cases, 'method': method,
        'optimizer_updates': steps, 'lr_step_updates': lr_step,
        'validation_every_updates': 50 if stage != 'legacy_anchor' else 10,
        'initial_lr': 0.0001, 'lr_gamma': 0.5, 'auxiliary_warmup_updates': 50,
        'early_stopping': False, 'training_split_manifest': 'UNRESOLVED',
        'evaluation_split_manifest': 'UNRESOLVED', 'source_manifest': 'UNRESOLVED',
        'status': 'PLANNED_NOT_LAUNCHABLE',
    })


for seed in (0, 1, 2):
    for steps, decay in ((4000, 1060), (8000, 2120)):
        for method in ('dice', 'bands', 'global_fg_bce', 'dice_ce_calibrated'):
            add('development_horizon', 'D0_historical', 0, seed, 10, method, steps, decay)
    for method in ('dice', 'bands'):
        add('legacy_anchor', 'D0_historical', 0, seed, 10, method, 750, 200)
    add('development_ce_sensitivity', 'D0_historical', 0, seed, 10, 'dice_ce_unit', 8000, 2120)
for method in ('dice', 'bands', 'global_fg_bce', 'dice_ce_calibrated', 'dice_ce_unit'):
    add('development_fraction_check', 'D0_historical', 0, 0, 52, method, 8000, 2120)

for split in ('R0', 'R1', 'R2'):
    for subset in (101, 202):
        for seed in (11, 22):
            for method in METHODS:
                add('replication_10', split, subset, seed, 10, method, 8000, 2120)
            for method in ('dice', 'bands'):
                add('fraction_52', split, subset, seed, 52, method, 8000, 2120)
            add('boundary_placement_control', split, subset, seed, 10, 'random_matched_fg_bce', 8000, 2120)

assert len(matrix) == 122 and len({r['run_id'] for r in matrix}) == len(matrix)
counts = Counter(r['stage'] for r in matrix)
assert counts == {'development_horizon': 24, 'legacy_anchor': 6,
                  'development_ce_sensitivity': 3, 'development_fraction_check': 5,
                  'replication_10': 48, 'fraction_52': 24, 'boundary_placement_control': 12}
assert all(r['optimizer_updates'] % r['validation_every_updates'] == 0 for r in matrix)

# All methods receive the same conservative per-update estimate; novel objectives
# and the new evaluation cadence must be benchmarked before this becomes a cap.
t_update = max(x['seconds_per_update'] for x in observations)
t_val = max(x['seconds_per_52_case_validation'] for x in observations)
t_checkpoint = max(x['seconds_per_checkpoint_event'] for x in observations)
assumptions = {
    'gpu': 'A100 MIG 3g.40gb, same environment as archived timings',
    'seconds_per_update': t_update,
    'seconds_per_52_case_validation': t_val,
    'seconds_per_checkpoint_event': t_checkpoint,
    'checkpoint_multiplier_per_validation': 2,
    'startup_and_final_evaluation_seconds_per_production_run': 120,
    'shared_calibration_seconds_per_subset_seed_fraction': 120,
    'reserve_multiplier': 1.25,
    'reference_validation_case_count': 52,
    'limitations': ['Novel loss timing unmeasured', 'Subject grouping and validation sizes unresolved',
                   'New update-based loop not implemented', 'Queue time excluded',
                   'Storage/input startup may differ', 'No statistical power guarantee'],
}
stage_seconds = Counter()
for r in matrix:
    checks = r['optimizer_updates'] // r['validation_every_updates']
    seconds = (r['optimizer_updates'] * t_update + checks * (t_val + 2*t_checkpoint) + 120)
    r['estimated_production_slice_seconds'] = seconds
    stage_seconds[r['stage']] += seconds
calibration_blocks = {(r['split_role'], r['subset_seed'], r['model_seed'], r['training_case_count'])
                      for r in matrix if r['method'] != 'dice'}
assert len(calibration_blocks) == 28
shared_cal_seconds = len(calibration_blocks) * 120
nominal = sum(stage_seconds.values()) + shared_cal_seconds
plan = {
    'version': 'draft-1', 'date': '2026-09-19', 'authorization': 'Planning only; do not submit jobs',
    'proposed_budget_slice_hours': [30, 50], 'training_jobs_submitted': 0,
    'source_audit': str(AUDIT), 'observed_timing_summaries': observations,
    'budget_assumptions': assumptions, 'production_runs_by_stage': dict(counts),
    'shared_calibration_source_runs': len(calibration_blocks),
    'stage_production_slice_hours': {k:v/3600 for k,v in stage_seconds.items()},
    'shared_calibration_slice_hours': shared_cal_seconds/3600,
    'total_slice_hours_nominal': nominal/3600,
    'total_slice_hours_with_reserve': nominal*1.25/3600,
    'four_arm_plan_slice_hours_with_reserve': (nominal-stage_seconds['boundary_placement_control'])*1.25/3600,
    'reduced_27_run_development_slice_hours_with_reserve':
        (stage_seconds['development_horizon']+stage_seconds['development_ce_sensitivity']+3*120)*1.25/3600,
    'primary_contrast': 'bands minus global_fg_bce at 10 cases, development-selected checkpoint on held-out evaluation',
    'supporting_contrasts': ['bands minus dice', 'bands minus dice_ce_selected', 'bands minus random_matched_fg_bce'],
    'primary_speed_target_foreground_dice': 0.78,
    'secondary_speed_targets_foreground_dice': [0.76, 0.80],
    'sustained_target_consecutive_evaluations': 3,
    'proposed_practical_margin_dice_percentage_points': 0.5,
    'margin_status': 'Scientific planning threshold; not clinically validated or a power claim',
    'unresolved_before_execution': ['Verified case-to-subject mapping or explicitly case-level scope',
        'Actual disjoint training/development/evaluation manifests',
        'Independent external or demonstrably untouched evaluation availability',
        'Frozen source; current working tree contains unrelated user edits',
        'Update-based loop, global BCE and general calibration implementation',
        'Preflight timings, storage capacity and chosen CE policy'],
}
(HERE/'PLAN.json').write_text(json.dumps(plan, indent=2)+'\n')
with (HERE/'RUN_MATRIX.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(matrix[0]))
    w.writeheader()
    w.writerows(matrix)
budget = ['# Proposed compute budget', '',
          'Generated from archived timing logs plus explicit planning allowances. These are allocated MIG-slice hours, not full-GPU hours, elapsed calendar time, or measured future costs.', '',
          '| Stage | Production runs | Estimated slice-hours |', '|---|---:|---:|']
for k,v in stage_seconds.items():
    budget.append(f'| {k} | {counts[k]} | {v/3600:.2f} |')
budget += [f'| Shared calibration sources | {len(calibration_blocks)} | {shared_cal_seconds/3600:.2f} |',
           f'| Total before reserve | {len(matrix)} production + {len(calibration_blocks)} source | {nominal/3600:.2f} |',
           f'| Total with 25% reserve | same matrix | {nominal*1.25/3600:.2f} |', '',
           f'Four-arm plan without the additional matched-random control: {(nominal-stage_seconds["boundary_placement_control"])*1.25/3600:.2f} slice-hours including reserve. The matched-random control isolates boundary placement from merely supervising fewer voxels; decide whether to include it before replication results are revealed.', '',
           f'Reduced development-only plan: 27 production runs (24 horizon runs and three unit-CE checks) plus three calibration sources, {(stage_seconds["development_horizon"]+stage_seconds["development_ce_sensitivity"]+3*120)*1.25/3600:.2f} slice-hours including reserve. This omits historical anchors, the 52-case check and all replication.', '',
           f'Formula per production run: updates × {t_update:.5f} s + validation events × ({t_val:.3f} s + 2 × {t_checkpoint:.3f} s) + 120 s startup/final-evaluation allowance.', '',
           'Calibration: 120 seconds per unique split/subset/model-seed/fraction shared within the experiment, with no validation-based weight selection. This is an allowance, not a measured calibration duration.', '',
           'A standalone-method time-to-target must charge its full calibration-source and gradient-calibration time, even if the study reuses a cached source among methods. Research cost and standalone method cost are different accounting views.', '',
           'If validation contains N case files, scale the validation term provisionally by N/52; measure it before launch. The budget assumes an update-based loop with validation every 50 updates. The current epoch-based runner is not equivalent and would have a different budget.', '',
           'Rebenchmark a small preflight before committing to the full matrix. Do not spend the reserve selectively on unfavorable seeds. A budget reduction should remove a complete future stage or balanced blocks, not stop arms based on their results.', '',
           'No job has been submitted. RUN_MATRIX.csv is intentionally non-launchable until manifests and source hashes are resolved.']
(HERE/'BUDGET.md').write_text('\n'.join(budget)+'\n')
print(json.dumps({'production_runs':len(matrix), 'calibration_sources':len(calibration_blocks),
                  'slice_hours_nominal':nominal/3600,'slice_hours_with_reserve':nominal*1.25/3600,
                  'stage_counts':dict(counts)},indent=2))
