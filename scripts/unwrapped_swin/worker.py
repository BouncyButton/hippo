"""Run the complete study sequentially inside exactly one GPU allocation."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
from scripts.depth_supervision.utils import atomic_json, digest
from .models import ARMS


def finish_stage(root, state, stage, returncode, seconds):
    state['stages'].append(dict(stage=stage, returncode=returncode, seconds=seconds))
    if returncode:
        state.update(state='failed', failed_stage=stage, finished_unix=time.time())
    atomic_json(root / 'status.json', state)
    if returncode:
        raise RuntimeError(f'Stage {stage} failed; stopped without skipping experiments. See logs/{stage}.log')


def verify(root):
    source = root / 'source'
    p = json.loads((source / 'docs/experiments/unwrapped_swin_20261006/protocol.json').read_text())
    for name, expected in json.loads((root / 'payload_manifest.json').read_text()).items():
        if digest(root / name) != expected:
            raise ValueError('Staged payload changed: ' + name)
    for name, expected in p['source_sha256'].items():
        if digest(source / name) != expected:
            raise ValueError('Source changed: ' + name)
    if p.get('native_training_identity_sha'):
        previous = json.loads((root / 'prior_protocol.json').read_text())
        if previous['identity_sha'] != p['native_training_identity_sha']:
            raise ValueError('Native training identity does not match prior protocol')
        for key, value in previous.items():
            if key not in ('name', 'identity_sha', 'source_sha256') and p[key] != value:
                raise ValueError('Native training setting changed during repair: ' + key)
        for name in previous['source_sha256']:
            if name.startswith(('scripts/depth_supervision/', 'baselines/')) or name == 'scripts/unwrapped_swin/models.py':
                if digest(source / name) != previous['source_sha256'][name]:
                    raise ValueError('Native model or objective code changed: ' + name)
    if digest(p['pkl']) != p['pkl_sha256']:
        raise ValueError('Dataset changed')
    for package, version in p['required_dependencies'].items():
        if importlib.metadata.version(package) != version:
            raise ValueError('Dependency version changed: ' + package)
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Run only inside an allocated bf16 CUDA Slurm job')
    atomic_json(root / 'environment.json', dict(job_id=os.environ['SLURM_JOB_ID'],
        torch=torch.__version__, gpu=torch.cuda.get_device_name(),
        packages={k: importlib.metadata.version(k) for k in ['monai', 'libigl', 'trimesh', 'scikit-image', 'scipy', 'numpy']},
        protocol_sha256=p['identity_sha']))
    return p


def cuda_preflight(root, fraction):
    from .test_pipeline import real_model_smoke
    if fraction == 'synthetic':
        from .geometry import build_atlas
        xyz = np.indices((64, 64, 64)).transpose(1, 2, 3, 0) - 32
        mask = ((xyz / [8., 16., 8.]) ** 2).sum(-1) < 1
        atlas, _ = build_atlas(np.random.default_rng(0).normal(size=mask.shape).astype(np.float32),
                               mask, np.eye(4), np.ones_like(mask))
        name = 'synthetic'
    else:
        manifest = json.loads((root / 'atlas' / fraction / 'manifest.json').read_text())
        # Largest chart workload exercises the most expensive path. Discard all smoke weights.
        largest = max(manifest['reports'], key=lambda r: sum(np.prod(x['grid_shape']) for x in r['charts_audit'] if x['view'] == 'unfold'))
        name = largest['case']
        with np.load(root / 'atlas' / fraction / (name + '.npz')) as archive:
            atlas = {k: archive[k] for k in archive.files}
    result = real_model_smoke(atlas, 'cuda')
    result.update(case=name, peak_memory_bytes=torch.cuda.max_memory_allocated())
    atomic_json(root / f'cuda_branch_preflight_{fraction}.json', result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--cuda-preflight', choices=['synthetic', '5pct', 'full'])
    args = parser.parse_args()
    torch.set_num_threads(4)
    root = args.root
    if args.cuda_preflight:
        cuda_preflight(root, args.cuda_preflight)
        return
    p = verify(root)
    (root / 'logs').mkdir(exist_ok=True)
    state = dict(protocol_sha256=p['identity_sha'], job_id=os.environ['SLURM_JOB_ID'],
                 started_unix=time.time(), state='running', stages=[])

    def execute(stage, argv):
        state['current_stage'] = stage
        atomic_json(root / 'status.json', state)
        print('Starting ' + stage, flush=True)
        started = time.time()
        with (root / 'logs' / (stage + '.log')).open('a', buffering=1) as log:
            result = subprocess.run([sys.executable, '-u', '-m', *argv], cwd=root / 'source', stdout=log, stderr=subprocess.STDOUT)
        finish_stage(root, state, stage, result.returncode, time.time() - started)

    def train(fraction, arm, *extra):
        from .train import completed_record, run_identity
        guide_fold = int(extra[-1]) if extra else None
        output = root / 'guides' / fraction / str(guide_fold) if extra else root / 'runs' / fraction / arm
        suffix = '_guide' + extra[-1] if extra else ''
        stage = fraction + '_' + arm + suffix
        record = completed_record(output, run_identity(p, fraction, arm, guide_fold))
        if record is not None:
            state['stages'].append(dict(stage=stage, returncode=0, seconds=0, reused_completed=True,
                                        completed_epochs=record['completed_epochs']))
            atomic_json(root / 'status.json', state)
            print('Reusing completed ' + stage, flush=True)
            return
        return execute(fraction + '_' + arm + suffix,
                       ['scripts.unwrapped_swin.train', '--root', str(root), '--fraction', fraction, '--arm', arm, *extra])

    execute('geometry_unit_checks', ['scripts.unwrapped_swin.test_pipeline'])
    execute('cuda_preflight_synthetic', ['scripts.unwrapped_swin.worker', '--root', str(root), '--cuda-preflight', 'synthetic'])
    for fraction in ('5pct', 'full'):
        train(fraction, 'dice')
        train(fraction, 'depth')
        for k in (0, 1):
            train(fraction, 'dice', '--guide-fold', str(k))
        execute(fraction + '_atlas', ['scripts.unwrapped_swin.train', '--root', str(root), '--fraction', fraction, '--prepare-atlas'])
        execute(fraction + '_cuda_preflight', ['scripts.unwrapped_swin.worker', '--root', str(root), '--cuda-preflight', fraction])
        for arm, settings in ARMS.items():
            if settings:
                train(fraction, arm)
    try:
        from .report import report
        report(p, root)
    except Exception:
        state['report_error'] = traceback.format_exc()
    completed = sum((root / 'runs' / f / a / 'completion.json').exists() for f in ('5pct', 'full') for a in p['arms'])
    state.update(state='complete' if completed == p['main_runs'] and not state.get('report_error') else 'incomplete',
                 completed_main_runs=completed, finished_unix=time.time(), current_stage=None)
    atomic_json(root / 'status.json', state)
    if state['state'] != 'complete':
        raise RuntimeError('Some stages failed; inspect status.json and per-stage logs')


if __name__ == '__main__':
    main()
