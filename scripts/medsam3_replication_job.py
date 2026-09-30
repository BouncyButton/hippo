"""One Slurm allocation for all nine paired three-arm replication runs."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

from medsam3_preflight import digest, quota_free_bytes
from medsam3_replication import validate_study, required_quota_bytes


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def verify_finished(path, protocol):
    result = json.loads((path/'results.json').read_text())
    audit = json.loads((path/'audit.json').read_text())
    if result['protocol'] != protocol or result['status'] != 'complete' or audit['status'] != 'passed':
        raise RuntimeError('Completed run disagrees with the frozen study.')
    if audit.get('results_sha256') != digest(path/'results.json'):
        raise RuntimeError('Completed result changed after audit.')
    for arm in protocol['arms']:
        if digest(path/arm/'final_lora_weights.pt') != result['training'][arm]['final_adapter_sha256']:
            raise RuntimeError('Saved adapter integrity check failed.')
    for name, expected in result['output_hashes'].items():
        if digest(path/name) != expected:
            raise RuntimeError('Prediction integrity check failed.')


def verify_run_data(data, hashes, protocol):
    for case in protocol['train_cases'] + protocol['evaluation_cases']:
        for name in (f'imagesTr/{case}_0000.nii.gz', f'labelsTr/{case}.nii.gz'):
            if digest(data/name) != hashes[name]:
                raise RuntimeError(f'Frozen study data changed: {name}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'data'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--study', type=Path, default=Path('docs/experiments/medsam3_replication_20260928/protocols/study.json'))
    args = p.parse_args()
    root, data = args.root.resolve(), args.data.resolve()
    source = root/'source'
    study_path = source/args.study
    study = json.loads(study_path.read_text())
    job = os.environ['SLURM_JOB_ID']
    status_path = root/f'job_{job}_status.json'
    completed, current = [], None
    try:
        if sys.flags.optimize:
            raise RuntimeError('Assertions must be enabled for scientific audits.')
        folds = json.loads((data/'splits_final.json').read_text())
        validate_study(study, folds)
        for name, expected in study['data_hashes'].items():
            if digest(data/name) != expected:
                raise RuntimeError(f'Dataset changed: {name}')
        results = root/'results'
        results.mkdir(exist_ok=True)
        for name, protocol in study['protocols'].items():
            path = results/name
            if path.exists():
                verify_finished(path, protocol)
                completed.append(name)
        # Explicitly require RAM-backed scratch; different mount device IDs alone
        # do not establish that an arbitrary /tmp is outside the quota filesystem.
        local = Path('/dev/shm')
        filesystem = subprocess.check_output(['stat', '-f', '-c', '%T', str(local)], text=True).strip()
        if filesystem != 'tmpfs' or shutil.disk_usage(local).free < 8*1024**3:
            raise RuntimeError('Require at least 8 GiB of verified tmpfs scratch.')
        print(json.dumps(dict(scratch=str(local), filesystem=filesystem,
                              free_bytes=shutil.disk_usage(local).free)), flush=True)
        print(subprocess.check_output(['beegfs', 'quota', 'list-usage', '--uids', 'current',
                                       '--gids', 'current'], text=True), flush=True)
        with tempfile.TemporaryDirectory(prefix=f'medsam3-replication-{job}-', dir=local) as scratch:
            temporary = Path(scratch)
            base, ready = temporary/'sam3.pt', temporary/'ready.json'
            atomic_json(status_path, dict(status='waiting_for_base', job_id=job,
                         destination=str(base), ready=str(ready), completed_runs=completed))
            deadline = time.monotonic()+1200
            while not ready.is_file():
                if time.monotonic() > deadline:
                    raise TimeoutError('Checkpoint broadcast did not arrive.')
                time.sleep(2)
            expected = next(iter(study['protocols'].values()))['base_sha256']
            if json.loads(ready.read_text())['sha256'] != expected or digest(base) != expected:
                raise RuntimeError('Broadcast checkpoint hash mismatch.')
            for key, name in [('TMPDIR', 'tmp'), ('MPLCONFIGDIR', 'matplotlib'),
                              ('HF_HOME', 'hf'), ('HF_XET_CACHE', 'xet'),
                              ('TORCHINDUCTOR_CACHE_DIR', 'inductor'), ('TRITON_CACHE_DIR', 'triton')]:
                path = temporary/name
                path.mkdir()
                os.environ[key] = str(path)
            remaining = [n for n in study['protocols'] if n not in completed]
            minimum = required_quota_bytes(study, remaining)/1024**3
            subprocess.run([sys.executable, 'scripts/medsam3_preflight.py', '--root', str(root),
                            '--base-weights', str(base), '--minimum-free-gib', str(minimum)],
                           check=True, cwd=source)
            for current in remaining:
                verify_run_data(data, study['data_hashes'], study['protocols'][current])
                quota_output = subprocess.check_output(
                    ['beegfs', 'quota', 'list-usage', '--uids', 'current', '--gids', 'current'], text=True)
                pending = [n for n in remaining if n not in completed]
                required = required_quota_bytes(study, pending)
                if quota_free_bytes(quota_output) < required:
                    raise RuntimeError('Quota reserve insufficient for remaining frozen study.')
                atomic_json(status_path, dict(status='running', job_id=job, current_run=current,
                                             completed_runs=completed, total_runs=len(study['protocols'])))
                protocol_path = study_path.parent/f'{current}.json'
                if json.loads(protocol_path.read_text()) != study['protocols'][current]:
                    raise RuntimeError('Per-run protocol differs from study manifest.')
                calibration_args = []
                source_run = study['protocols'][current].get('calibration_source_run')
                if source_run and source_run != current:
                    if source_run not in completed:
                        raise RuntimeError('Shared training-only calibration source has not passed its audit.')
                    calibration_path = results/source_run/'calibration.json'
                    reference = json.loads((results/source_run/'results.json').read_text())['calibration']
                    if json.loads(calibration_path.read_text()) != reference:
                        raise RuntimeError('Shared calibration differs from audited source report.')
                    calibration_args = ['--calibration-input', str(calibration_path),
                                        '--calibration-sha256', digest(calibration_path)]
                subprocess.run([sys.executable, 'scripts/train_medsam3_constraint_few_shot.py',
                                '--upstream', str(root/'MedSAM3'), '--base-weights', str(base),
                                '--lora-weights', str(root/'weights/best_lora_weights.pt'),
                                '--data', str(data), '--protocol', str(protocol_path),
                                '--output', str(results/current), *calibration_args], cwd=source, check=True)
                subprocess.run([sys.executable, 'scripts/audit_medsam3_supervised.py',
                                '--results', str(results/current), '--data', str(data),
                                '--output', str(results/current/'audit.json')], cwd=source, check=True)
                verify_run_data(data, study['data_hashes'], study['protocols'][current])
                verify_finished(results/current, study['protocols'][current])
                completed.append(current)
                atomic_json(root/'completed_runs.json', dict(study_sha256=digest(study_path), runs=completed))
                fold = study['protocols'][current]['fold_index']
                fold_runs = [f'fold{fold}_seed{s}' for s in study['seeds']]
                if all(name in completed for name in fold_runs):
                    atomic_json(root/f'fold_completed_{fold}.json', dict(
                        status='complete', fold=fold, job_id=job, runs=fold_runs,
                        study_sha256=digest(study_path), completed_at_unix=time.time()))
            aggregate_args = ['--report-calibration-deviation'] if study.get('retains_prior_calibration_deviation') else []
            subprocess.run([sys.executable, 'scripts/medsam3_replication.py', 'aggregate',
                            '--study', str(study_path), '--results', str(results),
                            '--output', str(root/'analysis'), *aggregate_args], cwd=source, check=True)
        aggregate = json.loads((root/'analysis/aggregate.json').read_text())
        atomic_json(status_path, dict(status=aggregate['status'], job_id=job, completed_runs=completed,
                                     strict_study_gate_passed=aggregate['strict_study_gate_passed']))
    except Exception as exc:
        atomic_json(status_path, dict(status='failed', job_id=job, current_run=current,
                                     completed_runs=completed, error_type=type(exc).__name__, error=str(exc)))
        raise


if __name__ == '__main__':
    main()
