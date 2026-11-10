"""Freeze revised schedules while retaining the original measured training graphs."""
import json
import shutil

from campaign90 import ARMS, BUDGET, CAMPAIGN as OLD, MODELS, ROOT, digest, write_json
from optimization90 import CAMPAIGN, SCOPE, schedule_costs


def main():
    directory = ROOT / 'experiments' / CAMPAIGN
    reports = ROOT / 'reports' / CAMPAIGN
    reports.mkdir(parents=True, exist_ok=True)
    reference_path = ROOT / 'experiments/compute_budget_reference.json'
    snapshot = reports / 'previous_accounting_reference.json'
    if not snapshot.exists():
        shutil.copyfile(reference_path, snapshot)
    original_plan = json.loads((ROOT / 'experiments' / OLD / 'plan.json').read_text())
    protocols, schedules, parent_counts = {}, {}, {}
    for identity, spec in MODELS.items():
        old_dir = ROOT / 'experiments' / OLD / identity
        protocol = json.loads((old_dir / 'protocol.json').read_text())
        # Source-bound cost evidence is unchanged: only cap accounting changes.
        profile_path = ROOT / protocol['profile_evidence']['file']
        if digest(profile_path) != protocol['profile_evidence']['sha256']:
            raise ValueError('Profile evidence changed')
        costs = json.loads(profile_path.read_text())['costs']
        parent_counts[identity] = {a: s['max_steps'] for a, s in protocol['models'][spec['alias']]['objectives'].items()}
        objectives = {arm: schedule_costs(BUDGET, spec['baseline_flops'], costs[arm]['update_flops'],
                                         costs[arm]['calibration_flops']) for arm in ARMS}
        protocol.update(name=CAMPAIGN + '_' + identity, budget_scope=SCOPE,
                        parent_protocol_sha256=digest(old_dir / 'protocol.json'),
                        models={spec['alias']: {'parameters': spec['parameters'], 'objectives': objectives}},
                        calibration_policy=protocol['calibration_policy'].replace(
                            'All calibration charged within each run cap.',
                            'Calibration excluded from optimization cap; all attempts measured and reported separately.'),
                        accounting='Optimization cap includes model forward, objective loss, backward and optimizer arithmetic, including constraint losses. Calibration measured separately and excluded from cap. MAC=2. Validation, augmentation, target SDF preprocessing, I/O, integer/boolean/comparison operations and precision housekeeping excluded as before.',
                        restart_policy='Restart all24 runs from fresh initialization after user-requested cancellation/deletion of job685757. Keep original prefix/ramp and fixed LR horizons. Final optimization-budget Dice is the primary endpoint.')
        for name in ('scripts/optimization90.py', 'scripts/train_optimization90_core.py',
                     'scripts/train_optimization90.py'):
            protocol['training_source_sha256'][name] = digest(ROOT / name)
        model_dir = directory / identity
        model_dir.mkdir(parents=True, exist_ok=True)
        for name in ('splits_10patients.json', 'nnunet_architecture.json', 'split_provenance.json'):
            shutil.copyfile(old_dir / name, model_dir / name)
        write_json(model_dir / 'protocol.json', protocol)
        protocols[identity] = digest(model_dir / 'protocol.json')
        schedules[identity] = objectives
    plan = {**original_plan, 'name': CAMPAIGN, 'budget_scope': SCOPE,
            'protocol_sha256': protocols, 'schedules': schedules,
            'replaced_job_id': 685757, 'all_runs_fresh_initialization': True,
            'parent_plan_sha256': digest(ROOT / 'experiments' / OLD / 'plan.json'),
            'previous_max_steps': parent_counts,
            'additional_training_updates': sum(schedules[m][a]['max_steps'] - parent_counts[m][a]
                                               for m in MODELS for a in ARMS),
            'policy_selection': 'User revised accounting after observing calibration displaced updates. Exploratory results retained; use final optimization-budget endpoint and report calibration/total compute separately.'}
    write_json(directory / 'plan.json', plan)
    ref = json.loads(snapshot.read_text())
    ref.update(budget_scope=SCOPE, accounting=protocol['accounting'],
               schedules_path=str((directory / 'plan.json').relative_to(ROOT)),
               schedule_policy='Fixed baseline LR trajectories and unchanged calibration/prefix/ramp. Optimization-only cap includes constraint training cost; calibration excluded and reported separately.',
               accounting_change_snapshot=str(snapshot.relative_to(ROOT)),
               accounting_change_snapshot_sha256=digest(snapshot),
               policy_selection=plan['policy_selection'])
    write_json(reference_path, ref)
    rows = ['# Optimization-only cap: 31,976.45066709 GFLOPs', '',
            'Fold 0, seed 0, four models and six objectives. Ten training volumes, 52 validation volumes, 64³, batch 1. Calibration is separately measured and reported, not deducted from the optimization cap. Training constraint overhead remains inside the cap. Learning rates and calibration/prefix/ramp policies are unchanged; weights are recalibrated separately in each fresh constrained run. All24 runs restart from scratch in one allocation, as requested. The cancelled campaign metadata are archived before deleting its run outputs.', '',
            '| Model | Objective | Old updates | New updates | Epochs | Training GFLOPs | Calibration GFLOPs | Total GFLOPs |',
            '|---|---|---:|---:|---:|---:|---:|---:|']
    for m in MODELS:
        for a, s in schedules[m].items():
            rows.append(f'| {m} | {a} | {parent_counts[m][a]} | {s["max_steps"]} | {s["patient_pass_epochs"]:.1f} | {s["consumed_training_flops"]/1e9:.6f} | {s["calibration_flops"]/1e9:.6f} | {s["consumed_total_flops"]/1e9:.6f} |')
    rows.extend(['', 'This is equal optimization compute. Constrained arms spend additional calibration compute, so total compute is not equal. This policy was selected after inspecting development results and is recorded as such. Cancelled endpoints and source bindings are retained in a metadata archive.'])
    (directory / 'README.md').write_text('\n'.join(rows) + '\n')
    write_json(reports / 'prepared_plan.json', plan)
    print(json.dumps({m: {a: s['max_steps'] for a, s in v.items()} for m, v in schedules.items()}, indent=2))


if __name__ == '__main__':
    main()
