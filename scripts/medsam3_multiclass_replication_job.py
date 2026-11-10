"""Run one fold's three audited seeds; other folds may share the study concurrently."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

from medsam3_multiclass_replication import audit, digest, write
from medsam3_preflight import quota_free_bytes


def required_remaining(study, root):
    return sum(study['artifact_budget_bytes'][name] for name in study['protocols']
               if not (root/'results'/name/'audit.json').exists())/1024**3 + study['runtime_reserve_gib']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--fold', type=int, required=True)
    p.add_argument('--base-weights', type=Path,
                   help='Reuse the combined job checkpoint already verified in private tmpfs.')
    args = p.parse_args()
    root = args.root.resolve()
    study = json.loads((root/'study.json').read_text())
    assert args.fold in study['folds'] and not sys.flags.optimize
    job = os.environ['SLURM_JOB_ID']
    completed = []

    def status(stage, **fields):
        record = dict(status=stage, job_id=job, fold=args.fold, completed=completed, **fields)
        write(root/f'job_{job}_status.json', record)
        write(root/f'fold{args.fold}_status.json', record)

    try:
        source = root/'source'
        manifest = json.loads((source/'SOURCE_MANIFEST.json').read_text())
        for name, expected in manifest.items():
            assert digest(source/name) == expected, name
        assert subprocess.check_output(['stat', '-f', '-c', '%T', '/dev/shm'], text=True).strip() == 'tmpfs'
        assert shutil.disk_usage('/dev/shm').free >= 8*1024**3
        with tempfile.TemporaryDirectory(prefix=f'medsam3-ap-{job}-', dir='/dev/shm') as temporary:
            scratch = Path(temporary)
            os.chmod(scratch, 0o700)
            if args.base_weights:
                base = args.base_weights.resolve()
                assert base.is_relative_to(Path('/dev/shm')) and base.is_file()
            else:
                base, ready = scratch/'sam3.pt', scratch/'ready.json'
                status('waiting_for_base', destination=str(base), ready=str(ready), node=os.environ.get('SLURMD_NODENAME'))
                deadline = time.monotonic()+1800
                while not ready.exists():
                    if time.monotonic() > deadline:
                        raise TimeoutError('sbcast controller did not deliver checkpoint')
                    time.sleep(2)
            assert digest(base) == study['base_sha256']
            for key in ['TMPDIR', 'MPLCONFIGDIR', 'HF_HOME', 'HF_XET_CACHE', 'TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR']:
                folder = scratch/key
                folder.mkdir()
                os.environ[key] = str(folder)
            quota = subprocess.check_output(['beegfs', 'quota', 'list-usage', '--uids', 'current', '--gids', 'current'], text=True)
            minimum = required_remaining(study, root)
            write(root/f'job_{job}_quota_start.json', dict(raw=quota, free_gib=quota_free_bytes(quota)/1024**3, minimum_gib=minimum))
            if quota_free_bytes(quota)/1024**3 < minimum:
                raise RuntimeError('Insufficient global quota reserve for outstanding study outputs')
            subprocess.run([sys.executable, 'scripts/medsam3_preflight.py', '--root', str(root),
                            '--base-weights', str(base), '--minimum-free-gib', str(minimum)], cwd=source, check=True)
            subprocess.run([sys.executable, '-m', 'unittest', 'scripts.test_medsam3_multiclass',
                'scripts.test_medsam3_multiclass_replication', 'scripts.test_medsam3_supervised_constraints',
                'scripts.test_medsam3_calibration_cache'], cwd=source, check=True)
            for seed in study['seeds']:
                name = f'fold{args.fold}_seed{seed}'
                protocol = study['protocols'][name]
                path = root/'results'/name
                if path.exists():
                    raise RuntimeError(f'Output already exists; refusing overwrite: {name}')
                calibration_args = []
                source_name = protocol['calibration_source_run']
                if source_name != name:
                    reference_path = root/'results'/source_name
                    reference = json.loads((reference_path/'audit.json').read_text())
                    assert reference['status'] == 'passed'
                    assert reference['results_sha256'] == digest(reference_path/'results.json')
                    assert reference['calibration_sha256'] == digest(reference_path/'calibration.json')
                    calibration_args = ['--calibration-input', str(reference_path/'calibration.json'),
                                        '--calibration-sha256', reference['calibration_sha256']]
                status('running', current_run=name)
                subprocess.run([sys.executable, 'scripts/train_medsam3_multiclass.py',
                    '--upstream', str(root/'MedSAM3'), '--base-weights', str(base),
                    '--lora-weights', str(root/'weights/best_lora_weights.pt'),
                    '--data', study['data'], '--protocol', str(root/'protocols'/f'{name}.json'),
                    '--output', str(path), *calibration_args], cwd=source, check=True)
                status('auditing', current_run=name)
                audit(path, study['data'], root/'protocols'/f'{name}.json')
                completed.append(name)
            status('complete')
    except Exception as exc:
        status('failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
