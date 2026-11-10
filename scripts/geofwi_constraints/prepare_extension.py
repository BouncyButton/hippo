"""Freeze twelve confirmation runs and fifteen separately labelled explorations."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

from scripts.geofwi_cluster.campaign import CAP, read, write, sha
from scripts.geofwi_constraints.prepare import ROOT, replace

SOURCE = ROOT / 'experiments/geofwi3d_constraints_3fold_3seed_20261020'
PARENT = ROOT / 'experiments/geofwi3d_constraint_strength_20261020'
PARENT_REPORT = ROOT / 'reports' / PARENT.name
NAME = 'geofwi3d_strength_confirmation_exploration_20261021'
STAGE = ROOT / 'experiments' / NAME
REPORT = ROOT / 'reports' / NAME
REMOTE = Path('/mnt/beegfsstudents/home/3160552') / NAME
DESIGN = {
    'unetr_vit_b16': dict(arm='dice_bands_edge', confirmation='x4', exploration=['x6', 'x8'],
                         variants={'x4': [.4, 2.], 'x6': [.6, 3.], 'x8': [.8, 4.]}),
    'nnunet': dict(arm='dice_bands_sdf_depth', confirmation='x2', exploration=['x3', 'x4', 'x6'],
                   variants={'x2': [1.2, 1.5], 'x3': [1.8, 2.25], 'x4': [2.4, 3.], 'x6': [3.6, 4.5]})}


def prepare():
    assert not (STAGE / 'submission.json').exists(), 'Do not modify submitted stages'
    STAGE.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    inventory = read(PARENT / 'manifest.json')
    common = {'baseline_runtime.py', 'worker_entry.py', 'splits.json', 'data_manifest.json',
              'baseline_results.json', 'cpu_audit_basic.json', 'cpu_audit_unetr.json', 'run.sbatch'}
    for name, digest in inventory.items():
        if name.startswith('bundles/') or name in common:
            assert sha(PARENT / name) == digest, name
            dest = STAGE / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(PARENT / name, dest)
    assert sha(PARENT / 'campaign.py') == inventory['campaign.py']
    shutil.copyfile(PARENT / 'campaign.py', STAGE / 'parent_controller.py')
    shutil.copyfile(Path(__file__).with_name('extension_controller.py'), STAGE / 'campaign.py')
    if not (STAGE / 'data').exists():
        (STAGE / 'data').symlink_to((PARENT / 'data').resolve(), target_is_directory=True)
    replace(STAGE / 'run.sbatch', 'hippo-geofwi-strength', 'hippo-geofwi-strength-next')
    replace(STAGE / 'baseline_runtime.py', 'f"{lane_id}_{t[\'model\']}_{t[\'arm\']}_{t[\'strength\']}"',
            'f"{lane_id}_{t[\'model\']}_{t[\'arm\']}_{t[\'strength\']}_f{t[\'fold\']}"')
    replace(STAGE / 'baseline_runtime.py', "    for t in plan['tasks'][lane_id::2]:",
            "    by_id = {t['id']: t for t in plan['tasks']}\n    for identity in plan['lane_task_ids'][lane_id]:\n        t = by_id[identity]")
    proof = read(PARENT_REPORT / 'final_verification_20261021.json')
    assert proof['status'] == 'passed' and proof['verified_runs'] == 12 and proof['full_budget']
    controls = [r for r in proof['rows'] if r['strength'] == DESIGN[r['model']]['confirmation']]
    assert len(controls) == 6
    write(STAGE / 'controls.json', dict(rows=controls, parent_stage=str(REMOTE.parent / PARENT.name)))
    write(STAGE / 'reuse_verification.json', proof)
    write(STAGE / 'control_calibrations.json', read(PARENT_REPORT / 'selected_calibrations.json'))
    reused = [dict(r, role='confirmation', source_stage=str(REMOTE.parent / PARENT.name)) for r in controls]
    source_tasks = {t['id']: t for t in read(SOURCE / 'plan.json')['tasks']}
    source_inventory = read(SOURCE / 'manifest.json')
    tasks, differences = [], {}
    # Confirm both unseen folds before exploring on fold 0 within each GPU lane.
    blocks = [('confirmation', 1, None), ('confirmation', 2, None),
              ('exploration', 0, 0), ('exploration', 0, 1), ('exploration', 0, 2)]
    for role, fold, level in blocks:
        for seed in range(3):
            for model, design in DESIGN.items():
                if role == 'exploration' and level >= len(design['exploration']):
                    continue
                strength = design['confirmation'] if role == 'confirmation' else design['exploration'][level]
                target, cap = design['variants'][strength]
                arm = design['arm']
                origin_id = f'GeoFWI3D_{model}_{arm}_f{fold}s{seed}'
                task = deepcopy(source_tasks[origin_id])
                task.update(id=f'{origin_id}_strength_{strength}', strength=strength, role=role,
                            calibration_target_ratio=target, calibration_p95_cap=cap)
                control = next(r for r in controls if r['model'] == model and r['seed'] == seed)
                if role == 'exploration':
                    task['control_id'] = control['id']
                origin = SOURCE / 'protocols' / origin_id
                folder = STAGE / 'protocols' / task['id']
                folder.mkdir(parents=True, exist_ok=True)
                for file in origin.iterdir():
                    if file.is_file() and file.name != 'protocol.json':
                        assert sha(file) == source_inventory[str(file.relative_to(SOURCE))]
                        shutil.copyfile(file, folder / file.name)
                write(folder / 'dataset.json', dict(dataset_root=str(REMOTE / 'data'),
                                                    manifest=str(REMOTE / 'data/fold_0_manifest.json')))
                before = read(origin / 'protocol.json')
                after = deepcopy(before)
                after.update(name=NAME, baseline_run_id=task['id'], strength=strength, study_role=role,
                             data=str(REMOTE / 'protocols' / task['id'] / 'dataset.json'),
                             pkl_sha256=sha(folder / 'dataset.json'),
                             calibration_target_ratio=target, calibration_p95_cap=cap)
                if role == 'exploration':
                    after['constraint_control'] = control
                if model == 'unetr_vit_b16':
                    after['rescue_arm'] = task['id']
                diff = {k: dict(before=before.get(k), after=v) for k, v in after.items() if before.get(k) != v}
                assert set(diff) <= {'name', 'baseline_run_id', 'strength', 'study_role', 'data',
                                     'pkl_sha256', 'calibration_target_ratio', 'calibration_p95_cap',
                                     'constraint_control', 'rescue_arm'}
                selected = read(PARENT / 'protocols' / control['id'] / 'protocol.json')
                for field in ('models', 'initial_lrs', 'lr_horizon_updates', 'augmentation',
                              'sdf_policy', 'budget_flops', 'reference_initializations'):
                    assert after[field] == selected[field], (task['id'], field)
                if role == 'confirmation':
                    assert target == selected['calibration_target_ratio'] and cap == selected['calibration_p95_cap']
                differences[task['id']] = diff
                write(folder / 'protocol.json', after)
                tasks.append(task)
    write(STAGE / 'design.json', DESIGN)
    write(STAGE / 'protocol_differences.json', differences)
    write(STAGE / 'plan.json', dict(
        name=NAME, workers=2, cap=CAP, tasks=tasks, reused_rows=reused, total_cells=33,
        lane_task_ids=[[t['id'] for t in tasks if t['model'] == model] for model in DESIGN],
        new_runs=27, confirmation_new_runs=12, exploratory_new_runs=15,
        baseline_runs=0, baselines_reused=18, constraint_results_reused=6,
        parent_manifest_sha256=sha(PARENT / 'manifest.json'),
        selection='Confirmation strengths frozen before observing folds 1/2. Exploration uses fold 0 only; no adaptive selection from confirmation results.',
        stopping='All 27 fixed runs use full budgets, no halfway gates or replacement seeds; technical failures preserve recovery and stop continuation.',
        strength_definition='Multipliers apply to calibration target and p95 cap relative to the original constraints, not necessarily each final sequential coefficient.',
        storage=dict(minimum_free_gib=54, estimated_additional_peak_gib=46, laptop_required=False,
                     archive='archives/', policy='Verify cluster metadata archives before retiring completed bulky artifacts; preserve interrupted recovery'),
        continuation='Slurm self-submission at allocation boundary; no laptop collector or service'))
    write(STAGE / 'source_provenance.json', dict(parent_manifest_sha256=sha(PARENT / 'manifest.json'),
          original_suite_manifest_sha256=sha(SOURCE / 'manifest.json'),
          unchanged_training_sources={n: d for n, d in inventory.items() if n.startswith('bundles/') or n == 'worker_entry.py'}))
    print(json.dumps(dict(stage=str(STAGE), tasks=len(tasks), reused=len(reused), design=DESIGN)))


def freeze():
    assert not (STAGE / 'submission.json').exists()
    checks = [read(STAGE / f'strength_checks_{model}.json') for model in DESIGN]
    assert all(c['status'] == 'passed' and c['optimizer_updates'] == 0 for c in checks)
    plan = read(STAGE / 'plan.json')
    for check in checks:
        expected = {t['id'] for t in plan['tasks'] if t['model'] == check['model'] and t['role'] == 'exploration'}
        assert {r['task'] for r in check['calibration_replays']} == expected
        assert len(check['loss_checks']) == len(DESIGN[check['model']]['exploration'])
    test_result = read(STAGE / 'controller_checks.json')
    assert test_result['status'] == 'passed'
    write(STAGE / 'strength_checks.json', dict(status='passed', optimizer_updates=0, models=checks))
    write(STAGE / 'cpu_contract_checks.json', dict(status='passed', optimizer_updates=0,
          strength_checks_sha256=sha(STAGE / 'strength_checks.json'), controller_tests=test_result,
          inherited_geometry_and_objective_checks=read(SOURCE / 'cpu_contract_checks.json')))
    for name in ('prepare_extension.py', 'extension_controller.py', 'check_strength.py', 'test_extension.py'):
        shutil.copyfile(Path(__file__).with_name(name), STAGE / ('source_' + name))
    inventory = {str(p.relative_to(STAGE)): sha(p) for p in sorted(STAGE.rglob('*'))
                 if p.is_file() and 'data' not in p.relative_to(STAGE).parts
                 and '__pycache__' not in p.parts and p.name != 'manifest.json'}
    write(STAGE / 'manifest.json', inventory)
    subprocess.run([str(ROOT / '.venv/bin/python'), str(STAGE / 'campaign.py'), 'verify', '--stage', str(STAGE)], check=True)
    payload = REPORT / 'frozen_payload.tar.gz'
    with tarfile.open(payload, 'w:gz', compresslevel=1) as tar:
        for name in sorted(inventory):
            tar.add(STAGE / name, arcname=name, recursive=False)
        tar.add(STAGE / 'manifest.json', arcname='manifest.json', recursive=False)
    write(REPORT / 'preparation.json', dict(status='verified_ready_to_submit',
          manifest_sha256=sha(STAGE / 'manifest.json'), payload_sha256=sha(payload),
          payload_bytes=payload.stat().st_size, new_runs=27, reused_constraints=6, baseline_runs=0, laptop_required=False))
    for name in ('plan.json', 'design.json', 'strength_checks.json', 'controls.json', 'controller_checks.json'):
        shutil.copyfile(STAGE / name, REPORT / name)


if __name__ == '__main__':
    freeze() if '--freeze' in sys.argv else prepare()
