"""Read-only training observer; publish one small status file per minute."""
import argparse
from datetime import datetime, timezone
import json
import hashlib
from pathlib import Path
import subprocess
import time


def read_json(path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as error:
        return {'read_error': str(error)}


def observe(root):
    status = {'observed_at': datetime.now(timezone.utc).isoformat(), 'runs': {}}
    for seed, fold in ((seed, fold) for seed in range(3) for fold in range(3)):
        name = f'medsam3_f{fold}_s{seed}'
        output = root/'work_medsam3'/name
        history = read_json(output/'training_history.json')
        row = {'saved_updates': len(history) if isinstance(history, list) else 0}
        if isinstance(history, list) and history:
            row['latest_training'] = history[-1]
        for key in ('completion', 'training_summary', 'runtime_validation', 'worker_timing', 'result', 'compact_archive_receipt'):
            value = read_json(output/(key+'.json'))
            if value is not None:
                row[key] = value
        evaluation = read_json(output/'evaluation.json')
        if isinstance(evaluation, dict):
            row['evaluated_cases'] = len(evaluation.get('records', []))
            row['mean_dice'] = evaluation.get('mean_dice')
            row['case_scores'] = [{key: case[key] for key in ('case_name', 'split', 'macro_hard_dice')}
                                  for case in evaluation.get('records', [])]
            if row.get('completion', {}).get('status') == 'complete':
                row['evaluation_hash_verified'] = hashlib.sha256((output/'evaluation.json').read_bytes()).hexdigest() == row['completion']['evaluation_sha256']
        log = output/'worker.log'
        if log.exists():
            row['log_tail'] = log.read_text(errors='replace')[-(1000 if row.get('completion', {}).get('status') == 'complete' else 6000):]
        status['runs'][name] = row
    first = status['runs']['medsam3_f0_s0']
    done = first.get('completion', {})
    status['first_run_complete'] = done.get('status') == 'complete' and done.get('updates') == 200
    status['next_fold_training'] = status['runs']['medsam3_f1_s0']['saved_updates'] > 0
    status['completed_runs'] = sum(row.get('completion', {}).get('status') == 'complete' and
        row['completion'].get('updates') == 200 for row in status['runs'].values())
    status['archived_runs'] = sum(row.get('compact_archive_receipt', {}).get('status') == 'verified'
                                for row in status['runs'].values())
    controller_log = root/'medsam3_controller.log'
    status['controller_log_tail'] = controller_log.read_text(errors='replace')[-6000:] if controller_log.exists() else ''
    processes = []
    for process in Path('/proc').glob('[0-9]*'):
        try:
            command = (process/'cmdline').read_bytes().split(b'\x00')
            if b'scripts.colab_hfh.medsam_search' in command or b'scripts.colab_hfh.medsam' in command:
                processes.append(int(process.name))
        except OSError:
            pass
    status['active_processes'] = processes
    status['monitor_complete'] = (status['completed_runs'] == 9 and status['archived_runs'] == 9
        and not processes and 'exploration_complete' in status['controller_log_tail'])
    try:
        status['gpu'] = subprocess.check_output(['nvidia-smi',
            '--query-gpu=name,utilization.gpu,memory.used,memory.total',
            '--format=csv,noheader'], text=True, timeout=10).strip()
    except (OSError, subprocess.SubprocessError) as error:
        status['gpu_error'] = str(error)
    return status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('/content/hippo_hfh'))
    parser.add_argument('--hours', type=float, default=8)
    args = parser.parse_args()
    private = Path('/content/.hfh_storage_auth')
    command = [str(private/'rclone'), '--config', str(private/'rclone.conf'),
        '--tpslimit', '1', '--tpslimit-burst', '1', '--retries', '2',
        '--low-level-retries', '3', 'copyto']
    local = args.root/'medsam_live_monitor.json'
    remote = 'hfh_storage:results_medsam3_v1/live_monitor.json'
    deadline = time.monotonic()+args.hours*3600
    while time.monotonic() < deadline:
        status = observe(args.root)
        local.write_text(json.dumps(status, indent=2, allow_nan=False)+'\n')
        try:
            upload = subprocess.run(command+[str(local), remote], capture_output=True,
                                    text=True, timeout=120)
        except subprocess.TimeoutExpired:
            print(status['observed_at'], 'Status upload timed out; retrying next observation', flush=True)
            time.sleep(30)
            continue
        print(status['observed_at'], {k: v['saved_updates'] for k, v in status['runs'].items()},
              'upload', upload.returncode, flush=True)
        if upload.returncode:
            print(upload.stderr[-1200:], flush=True)
        if status['monitor_complete'] and upload.returncode == 0:
            return
        time.sleep(60)


if __name__ == '__main__':
    main()
