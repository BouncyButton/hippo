"""Run the explicitly authorized fold-0 baselines, with verified archival and resume."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', type=Path, required=True)
    ap.add_argument('--durable', type=Path, required=True)
    ap.add_argument('--model', choices=['unetr_vit_b16', 'basic_unet16'], required=True)
    ap.add_argument('--deadline', type=float, required=True)
    ap.add_argument('--verify-only', action='store_true')
    args = ap.parse_args()
    root, durable = args.bundle.resolve(), args.durable.resolve()
    sys.path.insert(0, str(root))
    import runtime as engine
    io = engine.io
    io.activate(root)
    engine.verify(root)
    unetr = args.model == 'unetr_vit_b16'
    if unetr:
        import lr3_runtime as campaign
        campaign.verify(root)
    seeds = [1, 2] if unetr else [0, 1, 2]
    durable.mkdir(parents=True, exist_ok=True)
    rows = []
    if args.verify_only:
        for seed in seeds:
            if unetr:
                campaign.protocol(root, f'lr6_s{seed}')
            else:
                engine.protocol(root, engine.task(seed, 0, args.model))
        print('Verified requested tasks:', args.model, seeds, flush=True)
        return
    value = io.binding(root, durable)
    if value['environment'] != io.read(root/'control_environment.json'):
        raise ValueError('Use the same A100-80GB software environment as the LR6 baseline')
    for seed in seeds:
        arm = f'lr6_s{seed}'
        t = campaign.task(arm) if unetr else engine.task(seed, 0, args.model)
        result = campaign.final_result(root, durable, arm) if unetr else engine.result(root, durable, t)
        io.write(durable/'status.json', dict(status='running', task=t, completed=rows))
        if result is None:
            if time.time() > args.deadline-180 or (durable.parent/'STOP').exists():
                io.write(durable/'status.json', dict(status='paused', task=t, completed=rows))
                sys.exit(75)
            if unetr:
                campaign.preflight(root, durable, arm)
                ok = campaign.execute(root, durable, arm, 414, args.deadline)
                if not ok:
                    sys.exit(75)
                result = campaign.final_result(root, durable, arm)
            else:
                folder, p = engine.protocol(root, t)
                # One replay diagnostic per architecture/bundle/environment, not per seed.
                expected = dict(status='passed', **value)
                receipt = durable/'preflight.json'
                if not receipt.exists() or io.read(receipt) != expected:
                    pf = root/'preflight/basic_unet16'
                    subprocess.run([sys.executable, str(root/'worker.py'), '--protocol-dir', str(folder),
                        '--output', str(pf), '--seed', '0', '--fold', '0', '--preflight'],
                        env=io.child_environment(root), check=True)
                    assert io.read(pf/'gpu_preflight.json')['status'] == 'passed'
                    io.copy_verified(pf/'gpu_preflight.json', durable/'preflight/basic_unet16.json')
                    io.write(receipt, expected)
                out = root/'work'/t['id']
                recovery = durable/'recovery'/t['id']
                if not (out/'checkpoint_latest.pt').exists():
                    io.restore_recovery(out, recovery)
                out.mkdir(parents=True, exist_ok=True)
                io.copy_verified(folder/'protocol.json', out/'protocol.json')
                io.copy_verified(folder/'splits_10patients.json', out/'splits.json')
                if shutil.disk_usage(root).free < 16*1024**3:
                    raise RuntimeError('Need 16 GiB runtime headroom')
                io.check_drive_headroom(durable)
                if not (out/'completion.json').exists():
                    env = io.child_environment(root)
                    env.update(HIPPO_COLAB_RECOVERY=str(recovery), HIPPO_DEADLINE=str(args.deadline-60))
                    cmd = [sys.executable, '-u', str(root/'worker.py'), '--protocol-dir', str(folder),
                        '--output', str(out), '--seed', str(seed), '--fold', '0']
                    print('RUN', t['id'], 'full budget: 952 updates', flush=True)
                    with (out/'train.log').open('a') as log, (out/'train.log').open() as progress:
                        progress.seek(0, 2)
                        proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
                        sent = False
                        try:
                            while proc.poll() is None:
                                print(progress.read(), end='', flush=True)
                                if (durable.parent/'STOP').exists() and not sent:
                                    proc.send_signal(signal.SIGUSR1)
                                    sent = True
                                time.sleep(2)
                            print(progress.read(), end='', flush=True)
                        except KeyboardInterrupt:
                            proc.send_signal(signal.SIGUSR1)
                            proc.wait()
                            sys.exit(75)
                    if proc.returncode == 75:
                        sys.exit(75)
                    if proc.returncode:
                        io.copy_verified(out/'train.log', durable/'failures'/t['id']/'train.log')
                        raise RuntimeError('Training failed; recovery preserved')
                engine.validate_output(root, out, t)
                io.archive_completed(root, out, durable, recovery, t)
                result = engine.result(root, durable, t)
        if not result:
            raise ValueError('Completed archive did not verify')
        rows.append(result)
        io.write(durable/'results.json', rows)
        print('RESULT', json.dumps(result), flush=True)
    io.write(durable/'status.json', dict(status='complete', completed=rows))


if __name__ == '__main__':
    main()
