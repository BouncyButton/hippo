"""Use the MSD admission/telemetry framework for the explicitly listed MNI tasks."""
import argparse
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from scripts.colab_medsam3.parallel import AdaptiveLimit, SweepLock, Worker, gpu_snapshot
from scripts.colab_medsam3.runtime import copy_verified, quota
from scripts.cross_dataset_campaign.common import digest, read, write


def progress(output):
    ledger = output/'checkpoint_ledger.json'
    step = read(ledger)['step'] if ledger.exists() else 0
    path = output/'evaluation.json'
    volumes = len(read(path)['records']) if path.exists() else 0
    phase = 'evaluating' if (output/'training_summary.json').exists() else 'training' if step else 'loading'
    return dict(step=step,volumes=volumes,phase=phase)


def verify_completed(durable):
    if not (durable/'completion.json').exists():
        return False
    done = read(durable/'completion.json')
    if done['status'] != 'complete' or digest(durable/'evaluation.json') != done['evaluation_sha256']:
        raise ValueError('Completed evaluation changed')
    if digest(durable/'protocol.json') != done['protocol_sha256']:
        raise ValueError('Completed protocol changed')
    return True


def run(root, durable, base, maximum=0, hours=12):
    bundle = root/'bundle'
    for name, expected in read(bundle/'manifest.json').items():
        if digest(bundle/name) != expected:
            raise ValueError('Bundle file changed: '+name)
    plan = read(bundle/'plan.json')
    tasks = plan['tasks']
    # This notebook authorizes one n=2 pilot, not an automatic seed/sample sweep.
    if tasks != [dict(id='MNI_MedSAM3_n2_f0s0_native200',arm='dice',subjects=2,fold=0,seed=0)]:
        raise ValueError('Unexpected experiment scope')
    output = root/'work'
    output.mkdir(parents=True,exist_ok=True)
    if verify_completed(durable):
        print('Verified completed two-subject pilot; no repeat training.',flush=True)
        return
    with SweepLock(root):
        quota(durable)
        if shutil.disk_usage(root).free < 8*1024**3:
            raise RuntimeError('Need8GiB local recovery headroom')
        tuner = AdaptiveLimit(maximum=min(maximum or len(tasks),len(tasks)))
        tuner.memory_per_worker = 24.
        gpu = gpu_snapshot()
        if not tuner.can_launch(gpu,[]):
            raise RuntimeError('Insufficient GPU/CPU/RAM headroom for the requested worker')
        env = dict(os.environ,MNI_DURABLE=str(durable),MNI_MANIFEST_SHA=digest(bundle/'manifest.json'),
                   MPLCONFIGDIR=str(root/'matplotlib_cache'),PYTHONUNBUFFERED='1')
        command = [sys.executable,'-m','scripts.colab_mni_medsam3.optimized',
            '--protocol',str(bundle/'protocol.json'),'--output',str(output),
            '--base-weights',str(base),'--durable',str(durable)]
        stop = False
        def request_stop(*_):
            nonlocal stop
            stop = True
        handlers = {s:signal.signal(s,request_stop) for s in (signal.SIGINT,signal.SIGTERM,signal.SIGUSR1)}
        deadline = time.monotonic()+hours*3600
        process = None
        try:
            with (output/'train.log').open('a') as log:
                process = subprocess.Popen(command,cwd=bundle/'source',env=env,stdout=log,stderr=subprocess.STDOUT)
                initial = progress(output)
                worker = Worker(tasks[0],process,log,time.monotonic(),initial,state=initial)
                last_report = -float('inf')
                try:
                    while process.poll() is None:
                        stop = stop or (durable/'STOP').exists() or time.monotonic()>=deadline-180
                        ready = output/'supervisor_ready.json'
                        if stop and ready.exists() and read(ready)['pid']==process.pid:
                            process.send_signal(signal.SIGUSR1)
                        now = time.monotonic()
                        if now-last_report>=30:
                            gpu = gpu_snapshot();worker.state=progress(output)
                            tuner.observe(now,gpu,[worker])
                            status = dict(time=time.time(),active_tasks=1,scheduled_tasks=len(tasks),
                                worker_limit=min(tuner.resource_limit(gpu),len(tasks)),gpu=gpu,
                                peak_gib=worker.peak_gib,**worker.state,stopping=stop)
                            write(durable/'gpu_status.json',status)
                            with (output/'telemetry.jsonl').open('a') as stream:
                                import json
                                stream.write(json.dumps(status)+'\n')
                            print(f"GPU {gpu['utilization']:.0f}% | VRAM {gpu['used_gib']:.1f}/{gpu['total_gib']:.1f} GiB | 1 scheduled worker | {worker.state}",flush=True)
                            if gpu['total_gib']-gpu['used_gib']<4 or gpu['host_free_gib']<4:
                                stop=True
                            last_report=now
                        time.sleep(5)
                finally:
                    # Do not leave a worker behind if telemetry/control fails.
                    while process.poll() is None:
                        ready=output/'supervisor_ready.json'
                        if ready.exists() and read(ready)['pid']==process.pid:
                            process.send_signal(signal.SIGUSR1)
                        time.sleep(5)
            for name in ('train.log','telemetry.jsonl'):
                if (output/name).exists(): copy_verified(output/name,durable/name)
            if process.returncode==75:
                raise RuntimeError('Clean stop saved to Drive. Resume the same Run cell.')
            if process.returncode:
                print((output/'train.log').read_text()[-8000:],flush=True)
                raise RuntimeError('Worker failed; all recovery retained')
            if not verify_completed(durable):
                raise RuntimeError('Worker exited without verified completion')
            print('Two-subject pilot complete. Run the results cell to verify/archive/clean.',flush=True)
        finally:
            for sig,handler in handlers.items(): signal.signal(sig,handler)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('root','durable','base'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--max-workers',type=int,default=0)
    p.add_argument('--hours',type=float,default=12)
    a=p.parse_args()
    if a.max_workers<0 or a.hours<=0:p.error('Invalid worker/session limits')
    run(a.root,a.durable,a.base,a.max_workers,a.hours)
