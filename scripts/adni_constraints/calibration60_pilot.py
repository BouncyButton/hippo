"""60% sequential calibration for all nnUNet ADNI constraint arms."""
import argparse
from copy import deepcopy
import csv
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

from verified_archive import read, digest, write, prepare as prepare_archive, retire

NAME = 'adni_nnunet_constraints_target60_20261013'
PREVIOUS = 'adni_nnunet_heavy_constraints_20261013'
PARENT = 'adni_constraints_noaug_5pct_fold0_seed0_20261013'
BANDS = 'adni_nnunet_bands_feedback_20261013'
REMOTE = Path('/mnt/beegfsstudents/home/3160552')
CAP = 31976450667090
TARGET = .6
P95_CAP = .75


def select_at_target(rows, arm, target=TARGET, cap=P95_CAP):
    """Retain the original sequential joint-gradient algebra for every term."""
    import compute_constraint_objectives as original
    from scripts.capped_depth_delay import objectives as combined
    old_original, old_depth = original.calibrated_weight, combined.calibrated_weight
    def calibrated(values):
        return old_original(values, target=target, cap=cap)
    original.calibrated_weight = combined.calibrated_weight = calibrated
    try:
        return combined.select_weights(rows, arm)
    finally:
        original.calibrated_weight, combined.calibrated_weight = old_original, old_depth


def quota():
    raw = subprocess.check_output(['beegfs', 'quota', 'list-usage', '--uids', 'current', '--pool', 'pool_home'], text=True)
    match = re.search(r'(\d+(?:\.\d+)?)GiB/(\d+(?:\.\d+)?)GiB', raw)
    if not match: raise ValueError('Personal quota unavailable')
    return dict(used_gib=float(match[1]), limit_gib=float(match[2]),
                remaining_gib=float(match[2])-float(match[1]), raw=raw)


def verify(stage):
    for name, sha in read(stage/'manifest.json').items():
        assert digest(stage/name) == sha, name
    plan = read(stage/'plan.json')
    parent = Path(plan['parent_stage'])
    assert digest(parent/'manifest.json') == plan['parent_manifest_sha256']
    for name, sha in read(parent/'manifest.json').items():
        assert digest(parent/name) == sha, name
    assert digest(Path(plan['dataset'])) == plan['dataset_sha256']
    return plan


def prepare(stage):
    root = Path.cwd()
    parent = root/'experiments'/PARENT
    assert not stage.exists(), 'Never overwrite a frozen experiment'
    stage.mkdir(parents=True)
    shutil.copyfile(__file__, stage/'pilot.py')
    shutil.copyfile(Path(__file__).with_name('verified_archive.py'), stage/'verified_archive.py')
    prior = read(parent/'protocols/nnunet/protocol.json')
    records = root/'reports'/PARENT/'comparison_20261013/results/nnunet'
    references, runs = {}, []
    for arm in prior['requested_arm_order']:
        original = records/arm/'fold0/seed0/run'
        references[arm] = dict(final_dice=read(original/'validation_latest.json')['summary']['hippocampus_hard_dice'],
            best_dice=read(original/'completion.json')['best_validation_hard_dice'],
            completion_sha256=digest(original/'completion.json'),
            validation_sha256=digest(original/'validation_latest.json'))
        if arm == 'dice': continue
        schedule = prior['models']['nnunet']['objectives'][arm]
        calibration = read(original/'calibration.json')
        weights, statistics = select_at_target(deepcopy(calibration['cases']), arm)
        directory = stage/'conditions'/arm/'protocols/nnunet'
        directory.mkdir(parents=True)
        for filename in ('profile.json', 'splits_10patients.json', 'nnunet_architecture.json'):
            shutil.copyfile(parent/'protocols/nnunet'/filename, directory/filename)
        protocol = {**prior, 'name': NAME+'_'+arm, 'requested_arm_order': [arm],
            'models': {'nnunet': {**prior['models']['nnunet'], 'objectives': {arm: schedule}}},
            'calibration_target_ratio': TARGET, 'calibration_p95_cap': P95_CAP,
            'pilot_controller_sha256': digest(stage/'pilot.py'),
            'calibration_policy': 'Training-only sequential bands -> original edge -> SDF -> depth, each targeting 0.6 of the preceding joint logit-gradient norm, p95 capped at 0.75. Calibrate after update 9.',
            'pilot_change': 'Only calibration strength changes; original frozen timing, optimizer, data, no augmentation and optimization FLOPs cap.'}
        write(directory/'protocol.json', protocol)
        runs.append(dict(model='nnunet', alias='nnunet', objective=arm, updates=schedule['max_steps'],
            optimization_gflops=schedule['consumed_training_flops']/1e9,
            expected_weights=weights, expected_statistics=statistics,
            expected_anchor_model_sha256=calibration['anchor_model_sha256']))
    assert len(runs) == 13
    bands = root/'reports'/BANDS/'results/bands_target40'
    plan = dict(name=NAME, parent_stage=str(REMOTE/PARENT), parent_manifest_sha256=digest(parent/'manifest.json'),
        dataset=str(REMOTE/'hippo/datasets/Dataset103_ADNI/adni_hippocampus_full.pkl'), dataset_sha256=prior['pkl_sha256'],
        target_ratio=TARGET, p95_cap=P95_CAP, runs=runs, num_runs=13, fold=0, seed=0, data_fraction=.05,
        augmentation='none', budget_flops_per_run=CAP, calibration_in_cap=False,
        runtime_versions=read(parent/'plan.json')['runtime_versions'], references=references,
        reference_bands40=dict(final_dice=read(bands/'validation_latest.json')['summary']['hippocampus_hard_dice'],
            completion_sha256=digest(bands/'completion.json'), validation_sha256=digest(bands/'validation_latest.json')),
        previous_campaign=PREVIOUS, previous_job_id=read(root/'reports'/PREVIOUS/'submission.json')['job_id'],
        previous_manifest_sha256=digest(root/'experiments'/PREVIOUS/'manifest.json'),
        retention='Per-run durable local archive, SHA-256 acknowledgment, then delete both checkpoints before next run. Preserve all files on any failure.',
        storage=dict(minimum_quota_headroom_gib=6, conservative_peak_file_gib=2,
                     estimated_peak_quota_gib=4, reserve_gib=2, checkpoint_location='BeeGFS for interruption recovery'),
        comparison='13 fresh 60% conditions; reuse the unaugmented baseline and compare with original 10% and completed 40% results. The p95 ceiling scales from 0.5 to 0.75 to preserve the target-to-ceiling ratio.')
    write(stage/'plan.json', plan)
    remote = REMOTE/NAME
    (stage/'run.sbatch').write_text(f'''#!/usr/bin/env bash
#SBATCH --job-name=hippo-adni-nnunet-target60
#SBATCH --partition=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -Eeuo pipefail
export PYTHONPATH="{REMOTE/PARENT/'source'}:{REMOTE/PARENT/'source/scripts'}:{REMOTE/PARENT/'source/baselines/nnUNet'}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 nnUNet_compile=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export MPLCONFIGDIR=/tmp/hippo-adni-target60-mpl-${{SLURM_JOB_ID}}
/home/3160552/.conda/envs/hippocampus/bin/python "{remote/'pilot.py'}" --mode run --stage "{remote}"
''')
    write(stage/'manifest.json', {str(p.relative_to(stage)): digest(p) for p in stage.rglob('*') if p.is_file()})
    print(json.dumps(dict(prepared=str(stage), runs=len(runs), target=TARGET)), flush=True)


def train(arguments):
    from scripts.adni_constraints import train as entry
    from scripts.capped_depth_delay import core
    original_configure, original_calibrate = entry.configure, core.calibrate
    def configure(protocol, arm):
        assert protocol['augmentation'] == 'none' and protocol['calibration_target_ratio'] == TARGET and protocol['calibration_p95_cap'] == P95_CAP
        assert digest(Path(__file__)) == protocol['pilot_controller_sha256']
        original_configure(protocol, arm)
        core.select_weights = select_at_target
        def calibrate(*args, **kwargs):
            result = original_calibrate(*args, **kwargs)
            result.update(target_ratio=TARGET, p95_cap=P95_CAP, pilot_controller_sha256=digest(Path(__file__)))
            return result
        core.calibrate = calibrate
    entry.configure = configure
    sys.argv = [sys.argv[0], *arguments]
    entry.main()


def verify_run(stage, output, spec):
    """Read-only equivalent of the old verifier; archive must precede cleanup."""
    from scripts.capped_depth_delay.core import ARTIFACTS
    done, config = read(output/'completion.json'), read(output/'config.json')
    directory = stage/'conditions'/spec['objective']/'protocols/nnunet'
    protocol = read(directory/'protocol.json')
    split = read(directory/'splits_10patients.json')[0]
    schedule = protocol['models']['nnunet']['objectives'][spec['objective']]
    for key, name in ARTIFACTS.items(): assert digest(output/name) == done[key], name
    assert done['status'] == 'complete' and done['updates'] == spec['updates'] == done['training_attempts_charged']
    assert done['discarded_attempts'] == 0 and done['calibration_attempts'] == 1
    assert config['augmentation'] == 'none' and not config['smoke_test']
    assert config['model_identity'] == 'nnunet' and config['objective'] == spec['objective']
    assert config['fold'] == config['seed'] == 0 and config['protocol_sha256'] == digest(directory/'protocol.json')
    assert config['dataset_sha256'] == protocol['pkl_sha256']
    assert config['train_cases'] == split['train'] and config['validation_cases'] == split['val']
    assert done['consumed_training_flops'] == schedule['consumed_training_flops']
    assert done['calibration_flops'] == schedule['calibration_flops']
    assert 0 <= CAP-done['consumed_training_flops'] < schedule['flops_per_update']
    latest = read(output/'validation_latest.json')
    assert [r['case_name'] for r in latest['cases']] == split['val']
    with (output/'metrics.csv').open(newline='') as stream: history = list(csv.DictReader(stream))
    assert len(history) == 20 and int(history[-1]['step']) == spec['updates']
    assert float(history[-1]['macro_hard_dice']) == latest['summary']['hippocampus_hard_dice']
    assert max(float(r['macro_hard_dice']) for r in history) == done['best_validation_hard_dice']
    cal = read(output/'calibration.json')
    assert cal['target_ratio'] == TARGET and cal['p95_cap'] == P95_CAP and cal['anchor_successful_updates'] == 9
    assert cal['anchor_model_sha256'] == spec['expected_anchor_model_sha256']
    assert [r['case_name'] for r in cal['cases']] == split['train']
    assert set(cal['weights']) == set(spec['expected_weights'])
    for name, expected in spec['expected_weights'].items():
        assert abs(cal['weights'][name]-expected) <= max(1e-12, abs(expected)*1e-10), name
        assert cal['statistics'][name]['weighted_p95_ratio'] <= P95_CAP + 1e-12
    return dict(objective=spec['objective'], final_dice=latest['summary']['hippocampus_hard_dice'],
                best_dice=done['best_validation_hard_dice'], selected_update=done['selected_step'], updates=done['updates'],
                optimization_gflops=done['consumed_training_flops']/1e9, calibration_gflops=done['calibration_flops']/1e9,
                weights=cal['weights'], completion_sha256=digest(output/'completion.json'))


def run(stage):
    plan = verify(stage)
    import torch
    import monai
    assert torch.__version__ == plan['runtime_versions']['torch'] and monai.__version__ == plan['runtime_versions']['monai']
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    root = Path(str(stage)+'_results')
    root.mkdir(exist_ok=True)
    write(root/'launch.json', dict(job_id=os.environ['SLURM_JOB_ID'], gpu=torch.cuda.get_device_name(0), quota=quota()))
    rows = []
    for spec in plan['runs']:
        arm = spec['objective']
        output = root/arm
        assert not output.exists(), 'Refuse automatic replay or overwrite; interrupted runs require explicit recovery'
        assert quota()['remaining_gib'] >= plan['storage']['minimum_quota_headroom_gib']
        output.mkdir()
        print(json.dumps(dict(starting=arm)), flush=True)
        command = [sys.executable, str(stage/'pilot.py'), '--mode', 'train', '--model', 'nnunet',
                   '--objective', arm, '--fold', '0', '--seed', '0', '--pkl', plan['dataset'],
                   '--protocol-dir', str(stage/'conditions'/arm/'protocols/nnunet'),
                   '--output', str(output), '--threads', '4', '--resume']
        with (output/'training.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        row = verify_run(stage, output, spec)
        write(output/'verified_result.json', row)
        prepare_archive(output)
        print(json.dumps(dict(awaiting_durable_archive=arm)), flush=True)
        deadline = time.monotonic()+600
        while not (output/'archive_acknowledgment.json').exists():
            if time.monotonic() > deadline:
                raise TimeoutError('Archive acknowledgment absent: keep all checkpoints and stop sweep')
            time.sleep(5)
        cleanup = retire(output)
        row.update(cleanup=cleanup, gain_vs_baseline_pp=100*(row['final_dice']-plan['references']['dice']['final_dice']),
                   gain_vs_original_constraint_pp=100*(row['final_dice']-plan['references'][arm]['final_dice']))
        rows.append(row)
        write(root/'summary.json', dict(status='complete' if len(rows)==13 else 'in_progress', runs=rows,
              completed_runs=len(rows), total_runs=13, target_ratio=TARGET, remaining_quota=quota()))
        print(json.dumps(dict(completed=arm, final_dice=row['final_dice'], deleted_file_bytes=cleanup['released_file_bytes'])), flush=True)
    write(stage/'completion.json', dict(status='complete', runs=13, summary_sha256=digest(root/'summary.json')))


def submit(stage):
    assert os.getuid() == 1395 and stage.name == NAME
    with (stage/'submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (stage/'submission.json').exists():
            print((stage/'submission.json').read_text()); return
        plan = verify(stage)
        preflight = read(stage/'preflight.json')
        assert preflight['status'] == 'passed' and preflight['manifest_sha256'] == digest(stage/'manifest.json')
        usage = quota()
        assert usage['remaining_gib'] >= plan['storage']['minimum_quota_headroom_gib']
        jobs = [r.split('|') for r in subprocess.check_output(
            ['squeue', '-h', '-u', '3160552', '-o', '%i|%T|%j'], text=True).splitlines() if r.strip()]
        assert len(jobs) <= 1 and all(r[1] == 'RUNNING' for r in jobs), jobs
        dependencies = list(dict.fromkeys([str(plan['previous_job_id'])]+[r[0] for r in jobs]))
        command = ['sbatch', '--parsable', '--dependency=afterany:'+':'.join(dependencies)]
        command.append(str(stage/'run.sbatch'))
        job = subprocess.check_output(command, cwd=stage, text=True).strip().split(';')[0]
        assert job.isdigit()
        record = dict(job_id=int(job), runs=13, target_ratio=TARGET, augmentation='none', fold=0, seed=0,
            budget_gflops_per_run=CAP/1e9, dependency_job_ids=dependencies, quota_before_submission=usage,
            submitted_at_unix=time.time(), manifest_sha256=digest(stage/'manifest.json'))
        write(stage/'submission.json', record)
        print(json.dumps(record), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['prepare', 'train', 'run', 'submit'], required=True)
    parser.add_argument('--stage', type=Path)
    args, remaining = parser.parse_known_args()
    if args.mode == 'train': train(remaining)
    else:
        assert args.stage is not None and not remaining
        {'prepare': prepare, 'run': run, 'submit': submit}[args.mode](args.stage)
