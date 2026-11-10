"""One dependent GPU job running both profile variants at both data fractions."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import torch
from scripts.depth_supervision.utils import atomic_json, digest
from .train import completed_record, identity
from .report import report
from .views import freeze_branch_controls


def verify(root):
    p = json.loads((root/'protocol.json').read_text())
    for name, expected in json.loads((root/'payload_manifest.json').read_text()).items():
        if digest(root/name) != expected:
            raise ValueError('Payload changed: '+name)
    for path, expected in json.loads((root/'reference_manifest.json').read_text())['files'].items():
        if digest(path) != expected:
            raise ValueError('Frozen reference changed: '+path)
    if digest(p['pkl']) != p['pkl_sha256']:
        raise ValueError('Dataset changed')
    freeze_branch_controls(root, p)
    for package, version in p['required_dependencies'].items():
        if importlib.metadata.version(package) != version:
            raise ValueError('Package changed: '+package)
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Training requires allocated bf16 CUDA')
    atomic_json(root/'environment.json', dict(job_id=os.environ['SLURM_JOB_ID'], torch=torch.__version__,
        gpu=torch.cuda.get_device_name(), monai=importlib.metadata.version('monai'), protocol_sha256=p['identity_sha']))
    return p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    root = parser.parse_args().root
    torch.set_num_threads(4)
    (root/'logs').mkdir(exist_ok=True)
    state = dict(job_id=os.environ.get('SLURM_JOB_ID'), state='verifying', stages=[], started_unix=time.time())
    def execute(stage, argv):
        state['current_stage'] = stage
        atomic_json(root/'status.json', state)
        started = time.time()
        print('Starting '+stage, flush=True)
        with (root/'logs'/(stage+'.log')).open('a', buffering=1) as log:
            result = subprocess.run([sys.executable, '-u', '-m', *argv], cwd=root/'source', stdout=log, stderr=subprocess.STDOUT)
        state['stages'].append(dict(stage=stage, returncode=result.returncode, seconds=time.time()-started))
        atomic_json(root/'status.json', state)
        if result.returncode:
            raise RuntimeError(f'{stage} failed with exit code {result.returncode}; subsequent training stopped')
    try:
        p = verify(root)
        state.update(protocol_sha256=p['identity_sha'], state='running')
        execute('profile_unit_checks', ['scripts.tissue_profiles.tests', '--output', str(root/'unit_checks.json')])
        for entry in p['runs']:
            existing = completed_record(root/'runs'/entry['id'], identity(p, entry))
            if existing:
                state['stages'].append(dict(stage=entry['id'], returncode=0, reused_completed=True))
            else:
                execute(entry['id']+'_cuda_preflight', ['scripts.tissue_profiles.tests', '--root', str(root),
                    '--fraction', entry['fraction'], '--variant', entry['profile_variant'],
                    '--arm', entry['arm'],
                    '--output', str(root/(entry['id']+'_cuda_preflight.json'))])
                execute(entry['id'], ['scripts.tissue_profiles.train', '--root', str(root), '--run-id', entry['id']])
            report(p, root)
        result = report(p, root)
        if result['new_completed'] != p['main_runs']:
            raise RuntimeError('Missing completed run')
        state.update(state='complete', current_stage=None, completed_main_runs=p['main_runs'], finished_unix=time.time())
        atomic_json(root/'status.json', state)
    except Exception as error:
        state.update(state='failed', error=repr(error), finished_unix=time.time())
        atomic_json(root/'status.json', state)
        raise


if __name__ == '__main__':
    main()
