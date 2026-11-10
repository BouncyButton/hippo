"""One allocation, common prefix, calibrated finite screen, durable reports."""
import argparse
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import torch
from scripts.depth_supervision.utils import atomic_json, digest
from .matrix import experiment_matrix
from .train import completed_record, identity
from .warmup import prepare_prefix, seed_run
from .report import report
from .retention import retain_checkpoints, quota_free


def verify(root):
    p = json.loads((root/'protocol.json').read_text())
    for name, expected in json.loads((root/'payload_manifest.json').read_text()).items():
        if digest(root/name) != expected:
            raise ValueError('Payload changed: '+name)
    if digest(p['pkl']) != p['pkl_sha256']:
        raise ValueError('Dataset changed')
    assert p['runs'] == experiment_matrix() and p['main_runs'] == len(p['runs'])
    assert set(p['splits']) == {'5pct'} and p['fold'] == p['seed'] == 0
    assert len(p['splits']['5pct']['train']) == 10 and len(p['splits']['5pct']['val']) == 52
    for package, version in p['required_dependencies'].items():
        if importlib.metadata.version(package) != version:
            raise ValueError('Dependency changed: '+package)
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Require an allocated bf16 CUDA device')
    atomic_json(root/'environment.json', dict(job_id=os.environ['SLURM_JOB_ID'], torch=torch.__version__,
        gpu=torch.cuda.get_device_name(), monai=importlib.metadata.version('monai'), protocol_identity=p['identity_sha']))
    return p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    root = parser.parse_args().root
    torch.set_num_threads(4)
    (root/'logs').mkdir(exist_ok=True)
    with (root/'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = dict(job_id=os.environ.get('SLURM_JOB_ID'), state='verifying', stages=[], started_unix=time.time())
        def execute(stage, argv):
            state['current_stage'] = stage
            atomic_json(root/'status.json', state)
            print('Starting '+stage, flush=True)
            start = time.monotonic()
            with (root/'logs'/(stage+'.log')).open('a', buffering=1) as log:
                result = subprocess.run([sys.executable, '-u', '-m', *argv], cwd=root/'source', stdout=log, stderr=subprocess.STDOUT)
            state['stages'].append(dict(stage=stage, returncode=result.returncode, seconds=time.monotonic()-start))
            atomic_json(root/'status.json', state)
            if result.returncode:
                raise RuntimeError(stage+' failed; inspect log')
        try:
            p = verify(root)
            prepare_prefix(p, root)
            state['state'] = 'running'
            execute('cuda_preflight', ['scripts.band_evolution.tests', '--mode', 'model', '--device', 'cuda',
                    '--root', str(root), '--output', str(root/'cuda_preflight.json')])
            gate = root/'calibration_preflight.json'
            if not gate.exists():
                execute('calibration_preflight', ['scripts.band_evolution.calibration', '--root', str(root)])
            calibration = json.loads(gate.read_text())
            if not calibration['passed'] or calibration['objectives_checked'] != p['main_runs']:
                raise ValueError('Incomplete calibration gate')
            for entry in p['runs']:
                path = root/'calibration'/entry['id']/'record.json'
                if digest(path) != calibration['artifacts'][entry['id']]:
                    raise ValueError('Frozen calibration changed')
                if completed_record(root/'runs'/entry['id'], identity(p, entry)) is None:
                    if quota_free() < 1.:
                        raise RuntimeError('Less than 1 GiB quota free; completed results preserved, resume after space is available')
                    seed_run(p, root, entry)
                    execute(entry['id'], ['scripts.band_evolution.train', '--root', str(root), '--run-id', entry['id']])
                result = report(p, root)
                retain_checkpoints(p, root, result)
                state['completed_runs'] = result['completed']
                atomic_json(root/'status.json', state)
            if result['completed'] != p['main_runs']:
                raise RuntimeError('Incomplete campaign')
            state.update(state='complete', current_stage=None, finished_unix=time.time())
            atomic_json(root/'status.json', state)
        except Exception as error:
            state.update(state='failed', error=repr(error), finished_unix=time.time())
            atomic_json(root/'status.json', state)
            raise


if __name__ == '__main__':
    main()
