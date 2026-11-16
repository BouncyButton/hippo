"""One GPU worker, resumable finite exploration with seed-level rejection."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from scripts.colab_hfh.policy import cells, decision
from scripts.colab_hfh.runtime import read, write, digest, copy_verified, archive, verify_archive


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', required=True, choices=('baselines', 'medsam3'))
    for name in ('bundle', 'durable', 'work'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--hours', type=float, default=8)
    parser.add_argument('--candidate-count', type=int, default=32)
    args = parser.parse_args()
    if not 0 < args.hours <= 22 or not 3 <= args.candidate_count <= 32:
        parser.error('Use 0 < hours <= 22 and 3 <= candidate-count <= 32')
    args.work.mkdir(parents=True, exist_ok=True)
    args.durable.mkdir(parents=True, exist_ok=True)
    lock = (args.work/'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # One VM per notebook. A second VM cannot acquire an existing durable lease.
    lease = args.durable/'active_session'
    try:
        lease.mkdir()
    except FileExistsError:
        raise RuntimeError('An active/interrupted session lease exists. Stop the other VM, or audit its recovery before removing the lease.')
    stop = False
    def interrupt(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    deadline = time.time()+args.hours*3600
    accepted, rows, rejected = [], [], []
    process = None
    name = None
    write(lease/'owner.json', dict(pid=os.getpid(), started=time.time(), work=str(args.work)))
    try:
        for seed in range(args.candidate_count if args.kind == 'baselines' else 3):
            scores = {}
            for model, fold in cells(args.kind):
                if args.kind == 'baselines' and decision(args.kind, scores)[0] == 'rejected':
                    break
                name = f'{model}_f{fold}_s{seed}'
                output, durable = args.work/name, args.durable/'runs'/name
                protocol = args.bundle/'protocols'/model if args.kind == 'baselines' else args.bundle/'medsam_protocols'/f'f{fold}_s{seed}.json'
                if (durable/'archive_manifest.json').exists():
                    result = verify_archive(durable)
                    if (durable/'cleanup.json').exists():
                        rows.append(result); scores[(model, fold)] = result['validation_dice']; continue
                if stop or time.time() >= deadline or (args.durable/'STOP').exists():
                    write(args.durable/'search_status.json', dict(status='paused', accepted_seeds=accepted,
                        rejected_seeds=rejected, results=rows, next_cell=name))
                    return
                output.mkdir(parents=True, exist_ok=True)
                durable.mkdir(parents=True, exist_ok=True)
                from scripts.colab_adni.runtime import check_drive_headroom
                check_drive_headroom(args.durable)
                import psutil
                if psutil.virtual_memory().available < 12*1024**3:
                    raise RuntimeError('Less than 12 GiB available host RAM; use High-RAM')
                env = dict(os.environ, HIPPO_DEADLINE=str(deadline), MNI_MANIFEST_SHA=digest(args.bundle/'manifest.json'))
                command = [sys.executable, '-m', 'scripts.colab_hfh.'+('baseline' if args.kind == 'baselines' else 'medsam'),
                    '--output', str(output), '--durable', str(durable)]
                if args.kind == 'baselines':
                    command += ['--protocol-dir', str(protocol), '--pkl', str(args.bundle/'data_index.json'),
                                '--seed', str(seed), '--fold', str(fold)]
                else:
                    command += ['--protocol', str(protocol), '--base-weights', str(args.bundle.parent/'sam3.pt')]
                print('START', name, 'final Dice threshold 50%; no intermediate-loss pruning', flush=True)
                (output/'supervisor_ready.json').unlink(missing_ok=True)
                with (output/'worker.log').open('a') as log:
                    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
                    sent = False
                    last_report = 0.
                    while process.poll() is None:
                        if (stop or time.time() >= deadline or (args.durable/'STOP').exists()) and not sent:
                            if (output/'supervisor_ready.json').exists():
                                process.send_signal(signal.SIGUSR1); sent = True
                        if time.monotonic()-last_report >= 15:
                            last_report = time.monotonic()
                            try:
                                gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
                                    '--format=csv,noheader,nounits'], text=True).strip()
                            except (OSError, subprocess.CalledProcessError):
                                gpu = 'unavailable'
                            print(name, 'GPU util%, usedMiB,totalMiB:', gpu, flush=True)
                            if (output/'budget_ledger.json').exists():
                                print('charged updates:', read(output/'budget_ledger.json')['charged_attempts'], flush=True)
                            elif (output/'checkpoint_ledger.json').exists():
                                print('saved update:', read(output/'checkpoint_ledger.json')['step'], flush=True)
                        try:
                            process.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            pass
                copy_verified(output/'worker.log', durable/'worker.log')
                if process.returncode == 75:
                    write(args.durable/'search_status.json', dict(status='paused', accepted_seeds=accepted,
                        rejected_seeds=rejected, results=rows, next_cell=name))
                    print('Clean stop; verified recovery saved. Rerun to continue.', flush=True)
                    return
                if process.returncode:
                    raise RuntimeError(f'{name} failed, code {process.returncode}; not classified as a bad seed. Inspect {durable}/worker.log')
                result = archive(output, durable, args.kind, model, fold, seed, protocol)
                rows.append(result); scores[(model, fold)] = result['validation_dice']
                print('FINAL', name, f"{result['validation_dice']:.2%}", flush=True)
                write(args.durable/'search_status.json', dict(status='running', accepted_seeds=accepted,
                    rejected_seeds=rejected, results=rows))
            status, _ = decision(args.kind, scores)
            if status == 'accepted':
                accepted.append(seed)
            else:
                rejected.append(seed)
                print('REJECT', seed, 'remaining baseline cells skipped' if args.kind == 'baselines' else 'all three folds reported', flush=True)
            write(args.durable/'search_status.json', dict(status='running', accepted_seeds=accepted,
                rejected_seeds=rejected, results=rows, last_completed_seed=seed))
            if args.kind == 'baselines' and len(accepted) == 3:
                break
        status = ('complete' if len(accepted) == 3 else 'candidate_batch_exhausted') if args.kind == 'baselines' else 'exploration_complete'
        write(args.durable/'search_status.json', dict(status=status, accepted_seeds=accepted,
            rejected_seeds=rejected, results=rows, threshold=.5,
            warning='Validation-selected exploratory results; report every rejected seed, not an unbiased test estimate'))
        print(status, 'qualifying seeds:', accepted, flush=True)
    except Exception as error:
        write(args.durable/'search_status.json', dict(status='error', accepted_seeds=accepted,
            rejected_seeds=rejected, results=rows, next_cell=name,
            error_type=type(error).__name__, message=str(error)))
        raise
    finally:
        if process is not None and process.poll() is None:
            process.send_signal(signal.SIGUSR1)
            print('Waiting for active worker to finish its recovery upload', flush=True)
            while process.poll() is None:
                time.sleep(5)
        (lease/'owner.json').unlink(missing_ok=True)
        lease.rmdir()


if __name__ == '__main__':
    main()
