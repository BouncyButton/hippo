"""Durable orchestration for the original, update-matched MedSAM3 study."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time

ARMS = ('baseline', 'bands', 'sdf', 'bands_sdf', 'bands_edge',
        'bands_edge_sdf', 'depth', 'bands_depth', 'edge_depth', 'sdf_depth',
        'bands_edge_depth', 'bands_sdf_depth', 'edge_sdf_depth', 'bands_edge_sdf_depth')
EXISTING = ('baseline', 'bands', 'bands_edge')
FOLDS, SEEDS = (0, 1, 3), (17, 83, 191)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''): h.update(block)
    return h.hexdigest()


def relative(name):
    path = Path(name)
    if path.is_absolute() or '..' in path.parts: raise ValueError('Unsafe archive path')
    return path


def copy_verified(source, destination):
    destination = Path(destination); destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(destination.name + '.uploading')
    shutil.copyfile(source, temp)
    if digest(temp) != digest(source): raise IOError('Copy hash mismatch: ' + str(destination))
    temp.replace(destination)


def verify_bundle(root):
    for name, expected in read(root/'bundle_manifest.json').items():
        if digest(root/relative(name)) != expected: raise ValueError('Changed bundle file: ' + name)
    plan = read(root/'plan.json')
    expected = {(f, s, a) for f in FOLDS for s in SEEDS for a in ARMS if a not in ('bands', 'bands_edge')}
    identities = [(t['fold'], t['seed'], t['arm']) for t in plan['tasks']]
    if len(identities) != 108 or set(identities) != expected:
        raise ValueError('Expected exactly 99 missing comparisons and 9 fresh baselines; bands/bands+edge must not repeat')
    if (plan.get('calibration_policy') != 'fresh_runtime_sequential'
            or plan.get('historical_results_role') != 'reference_only'
            or any(t['arm'] != 'baseline' for t in plan['tasks'][:9])
            or any(t['id'] != f"MSD_MedSAM3_{t['arm']}_f{t['fold']}s{t['seed']}" for t in plan['tasks'])):
        raise ValueError('Matched study policy or controls-first schedule differs')
    imported = read(root/'existing_results.json')
    if len(imported['runs']) != 27 or any(r['audit_status'] != 'passed' for r in imported['runs']):
        raise ValueError('Missing original audited controls')
    for fold in FOLDS:
        for seed in SEEDS:
            p = read(root/'protocols'/f'fold{fold}_seed{seed}.json')
            required = dict(fold_index=fold, seed=seed, augmentation='none', updates=200,
                accumulation=2, warmup_updates=20, constraint_warmup_updates=20,
                learning_rate=5e-5, final_learning_rate=5e-6,
                gradient_ratio_target=.1, gradient_ratio_p95_cap=.5,
                calibration_seed=20260929, calibration_slabs_per_volume=4)
            if any(p.get(k) != v for k, v in required.items()): raise ValueError('Original recipe changed')
            if len(p['train_cases']) != 5 or set(p['train_cases']) & set(p['evaluation_cases']):
                raise ValueError('Invalid support split')
    for name, expected in read(root/'data_manifest.json').items():
        if digest(root/'data'/relative(name)) != expected: raise ValueError('Dataset differs: ' + name)
    if digest(root/'weights/sam3.pt') != plan['base_sha256']: raise ValueError('Base checkpoint differs')
    return plan


def activate(root):
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root/'source'))


def environment():
    import torch
    import psutil
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported() or torch.cuda.get_device_capability()[0] < 8:
        raise RuntimeError('This recipe requires native BF16 on A100/H100 or a compatible GPU')
    if torch.cuda.get_device_properties(0).total_memory < 38 * 1024**3:
        raise RuntimeError('Use an A100/H100 with at least 40 GB VRAM; the original training graph is retained')
    if psutil.virtual_memory().available < 12*1024**3:
        raise RuntimeError('Select a High-RAM Colab runtime; at least 12 GiB host RAM must be available before model loading')
    names = ('torch', 'torchvision', 'numpy', 'scipy', 'nibabel', 'timm', 'einops', 'Pillow')
    return dict(gpu=torch.cuda.get_device_name(), cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(), packages={n: importlib.metadata.version(n) for n in names})


def binding(root, durable):
    value = dict(bundle_sha256=digest(root/'bundle_manifest.json'), environment=environment())
    path = durable/'environment_binding.json'
    if path.exists() and read(path) != value:
        has_work = any((durable/'runs').glob('*/archive_manifest.json')) or any((durable/'recovery').glob('*/charged_ledger.json'))
        if has_work: raise ValueError('GPU/software changed after scientific work; use the original environment')
    write(path, value)
    return value


def quota(durable):
    import google.auth
    from googleapiclient.discovery import build
    credentials, _ = google.auth.default()
    record = build('drive', 'v3', credentials=credentials, cache_discovery=False).about().get(fields='storageQuota').execute()['storageQuota']
    write(durable/'storage_quota.json', dict(checked_at=time.time(), **record))
    if 'limit' in record and int(record['limit']) - int(record['usage']) < 8*1024**3:
        raise RuntimeError('Keep at least 8 GiB free on Drive, including trash in usage')


class RecoveryWriter:
    """Upload an immutable checkpoint snapshot without pausing GPU work."""
    def __init__(self, output, durable, interval=180):
        self.output, self.durable, self.interval = Path(output), Path(durable), interval
        self.last, self.future = -float('inf'), None
        self.pool = ThreadPoolExecutor(max_workers=1)

    def capture(self, force=False):
        if self.future:
            if force or self.future.done(): self.future.result()
            else: return
        if not force and time.monotonic()-self.last < self.interval: return
        checkpoint = self.output/'checkpoint_latest.pt'
        if not checkpoint.exists(): return
        snapshot = Path(tempfile.mkdtemp(prefix='snapshot-', dir=self.output.parent))
        os.link(checkpoint, snapshot/checkpoint.name)
        for name in ('protocol.json', 'checkpoint_ledger.json', 'calibration.json'):
            shutil.copyfile(self.output/name, snapshot/name)
        self.future = self.pool.submit(self.persist, snapshot)
        self.last = time.monotonic()

    def persist(self, snapshot):
        archive = snapshot.with_suffix('.tar')
        try:
            write(snapshot/'snapshot_manifest.json', {p.name: digest(p) for p in snapshot.iterdir()})
            with tarfile.open(archive, 'w') as tar:
                for path in snapshot.iterdir(): tar.add(path, arcname=path.name, recursive=False)
            sha = digest(archive); name = 'recovery-'+sha+'.tar'
            copy_verified(archive, self.durable/name)
            old = read(self.durable/'latest.json') if (self.durable/'latest.json').exists() else None
            write(self.durable/'latest.json', dict(file=name, sha256=sha))
            if old and old['file'] != name:
                p = self.durable/relative(old['file'])
                if p.exists() and digest(p) == old['sha256']: p.unlink()
        finally:
            archive.unlink(missing_ok=True); shutil.rmtree(snapshot)

    def close(self, force=False):
        try:
            if force: self.capture(force=True)
            if self.future: self.future.result()
        finally: self.pool.shutdown(wait=True)


def restore(output, recovery):
    marker = recovery/'latest.json'
    if not marker.exists():
        if (recovery/'charged_ledger.json').exists(): raise RuntimeError('Charged updates without a checkpoint; recovery audit required')
        return
    m = read(marker); archive = recovery/relative(m['file'])
    if digest(archive) != m['sha256']: raise ValueError('Recovery archive changed')
    output.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            relative(member.name)
            if not member.isfile(): raise ValueError('Recovery must contain regular files')
        tar.extractall(output, filter='data')
    for name, sha in read(output/'snapshot_manifest.json').items():
        if digest(output/relative(name)) != sha: raise ValueError('Recovery member changed')
    (output/'snapshot_manifest.json').unlink()


def check_ledger(output, recovery):
    charged = recovery/'charged_ledger.json'
    if charged.exists() and (not (output/'checkpoint_ledger.json').exists() or read(charged) != read(output/'checkpoint_ledger.json')):
        raise RuntimeError('A hard interruption lost charged updates. Checkpoint retained; do not silently replay them.')


def fold_calibration(root, durable, fold):
    folder = durable/'calibration'/f'fold{fold}'
    receipt = read(folder/'receipt.json')
    if (receipt['status'] != 'passed' or receipt['binding'] != read(durable/'environment_binding.json')
            or receipt['binding']['bundle_sha256'] != digest(root/'bundle_manifest.json')
            or receipt['calibration_sha256'] != digest(folder/'calibration.json')):
        raise ValueError('Shared calibration identity changed; rerun preflight before training')
    return read(folder/'calibration.json')


def completed(root, durable, task):
    folder = durable/'runs'/task['id']
    if not (folder/'cleanup.json').exists(): return False
    manifest = read(folder/'archive_manifest.json')
    if manifest['task'] != task or manifest['bundle_sha256'] != digest(root/'bundle_manifest.json'):
        raise ValueError('Archive identity differs')
    for name, sha in manifest['files'].items():
        if digest(folder/relative(name)) != sha: raise ValueError('Completed archive changed')
    return read(folder/'cleanup.json')['status'] == 'complete'


def archive_run(root, output, durable, recovery, task):
    done = read(output/'completion.json')
    if done['updates'] != 200 or done['status'] != 'complete': raise ValueError('Incomplete run')
    required = ('protocol.json', 'calibration.json', 'training_history.json', 'training_summary.json',
                'slab_schedule.json', 'runtime_recipe.json', 'evaluation.json', 'curves.png', 'completion.json', 'train.log')
    files = {name: digest(output/name) for name in required}
    target = durable/'runs'/task['id']
    for name in files: copy_verified(output/name, target/name)
    checkpoint = output/'checkpoint_latest.pt'
    checkpoint_sha = digest(checkpoint) if checkpoint.exists() else done['checkpoint_sha256']
    manifest = dict(task=task, files=files, bundle_sha256=digest(root/'bundle_manifest.json'),
                    checkpoint_sha256=checkpoint_sha)
    if not checkpoint.exists() and (not (target/'archive_manifest.json').exists() or read(target/'archive_manifest.json') != manifest):
        raise ValueError('Checkpoint missing without a previously verified archive')
    write(target/'archive_manifest.json', manifest)
    for name, sha in files.items():
        if digest(target/name) != sha: raise IOError('Archive readback failed')
    removed = []
    if checkpoint.exists():
        if digest(checkpoint) != manifest['checkpoint_sha256']: raise ValueError('Checkpoint changed before cleanup')
        removed.append(dict(file=str(checkpoint), bytes=checkpoint.stat().st_size, sha256=digest(checkpoint)))
        checkpoint.unlink()
    if (recovery/'latest.json').exists():
        m = read(recovery/'latest.json'); p = recovery/relative(m['file'])
        if p.exists():
            if digest(p) != m['sha256']: raise ValueError('Recovery changed before cleanup')
            removed.append(dict(file=str(p), bytes=p.stat().st_size, sha256=m['sha256']))
            p.unlink()
        (recovery/'latest.json').unlink()
    (recovery/'charged_ledger.json').unlink(missing_ok=True)
    write(target/'cleanup.json', dict(status='complete', removed=removed, verified_at=time.time()))


def report(root, durable):
    historical = read(root/'existing_results.json')['runs']
    rows = []
    for task in read(root/'plan.json')['tasks']:
        if completed(root, durable, task):
            record = read(durable/'runs'/task['id']/'completion.json')
            rows.append(dict(**task, **record, origin='Fresh matched Colab study'))
    baselines = {(r['fold'], r['seed']): r['validation_dice'] for r in rows if r['arm']=='baseline'}
    for row in rows:
        baseline = baselines.get((row['fold'], row['seed']))
        row['gain_pp'] = None if baseline is None else 100*(row['validation_dice']-baseline)
    aggregate = []
    for arm in ARMS:
        selected = [r for r in rows if r['arm']==arm]
        means = [statistics.mean(r['validation_dice'] for r in selected if r['fold']==f)
                 for f in FOLDS if sum(r['fold']==f for r in selected)==3]
        aggregate.append(dict(arm=arm, completed=len(selected), expected=0 if arm in ('bands', 'bands_edge') else 9,
            scope='historical_reference_only' if arm in ('bands', 'bands_edge') else 'fresh_matched',
            dice=100*statistics.mean(means) if len(means)==3 else None,
            gain_pp=statistics.mean(r['gain_pp'] for r in selected)
                if len(selected)==9 and all(r['gain_pp'] is not None for r in selected) else None))
    result = dict(runs=rows, aggregate=aggregate, new_completed=len(rows), new_total=108,
        historical_reference=historical, historical_results_used_as_matched_controls=False,
        budget='200 updates per run; not FLOP matched to the four-model sweeps',
        interpretation='Exploratory selected folds. Every primary comparison uses fresh controls with the same Colab environment and calibration. Historical cluster results are reference only. Same seeds do not guarantee bitwise training replay with nondeterministic production kernels.')
    write(durable/'comparison.json', result)
    print(f"Fresh matched experiments: {len(rows)}/108. Historical results: 27 reference only.", flush=True)
    for row in aggregate: print(row, flush=True)
    return result


def child(root, durable, command, *extra, deadline=None):
    env = os.environ.copy()
    env.update(PYTHONPATH=str(root/'source'), PYTHONUNBUFFERED='1', WANDB_MODE='disabled',
        MPLCONFIGDIR=str(root/'matplotlib_cache'))
    args = [sys.executable, str(root/'engine.py'), command, '--bundle', str(root), '--durable', str(durable), *map(str, extra)]
    if deadline: args += ['--deadline', str(deadline)]
    return args, env


def run(root, durable, max_runs, hours):
    plan = verify_bundle(root); value = binding(root, durable)
    if read(durable/'preflight.json') != dict(status='passed', **value): raise RuntimeError('GPU preflight required')
    deadline = time.time()+hours*3600; count = 0
    work = root/'work'; work.mkdir(exist_ok=True)
    with (work/'sweep.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for task in plan['tasks']:
            if completed(root, durable, task): continue
            if count >= max_runs or time.time()>deadline-180 or (durable/'STOP').exists(): break
            quota(durable)
            if shutil.disk_usage(root).free < 8*1024**3: raise RuntimeError('Keep 8 GiB free on runtime SSD')
            output = work/task['id']; recovery = durable/'recovery'/task['id']
            if not output.exists() or not any(output.iterdir()): restore(output, recovery)
            output.mkdir(exist_ok=True); check_ledger(output, recovery)
            if not (output/'completion.json').exists():
                args, env = child(root, durable, 'train', '--task', task['id'], deadline=deadline-120)
                print('RUN', task['id'], flush=True)
                with (output/'train.log').open('a') as log:
                    proc = subprocess.Popen(args, env=env, stdout=log, stderr=subprocess.STDOUT)
                    try:
                        while proc.poll() is None:
                            if (durable/'STOP').exists(): proc.send_signal(signal.SIGUSR1)
                            time.sleep(2)
                    except KeyboardInterrupt:
                        proc.send_signal(signal.SIGUSR1)
                        print('Saving recovery to Drive; wait for verification.', flush=True)
                        proc.wait(); return
                if proc.returncode == 75:
                    print('Clean stop saved. Rerun this cell to resume.'); return
                if proc.returncode:
                    copy_verified(output/'train.log', durable/'failures'/task['id']/'train.log')
                    raise RuntimeError((output/'train.log').read_text()[-6000:])
            archive_run(root, output, durable, recovery, task)
            count += 1; print('ARCHIVED', task['id'], flush=True)
            report(root, durable)
    report(root, durable)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('verify', 'reuse-preflight', 'preflight', 'run', 'report'))
    parser.add_argument('--bundle', type=Path, required=True); parser.add_argument('--durable', type=Path, required=True)
    parser.add_argument('--max-runs', type=int, default=108); parser.add_argument('--hours', type=float, default=22)
    parser.add_argument('--previous-results', type=Path); parser.add_argument('--previous-bundle', type=Path)
    args = parser.parse_args(); root = args.bundle.resolve(); durable = args.durable.resolve()
    durable.mkdir(parents=True, exist_ok=True); activate(root)
    if args.command == 'verify':
        verify_bundle(root); print('108 runs: 99 missing conditions plus 9 fresh baselines; bands and bands+edge are not repeated; 27 historical results for reference only; no augmentation; original 200-update recipe.')
    elif args.command == 'reuse-preflight':
        if args.previous_results is None or args.previous_bundle is None:
            parser.error('reuse-preflight requires --previous-results and --previous-bundle')
        verify_bundle(root)
        from reuse_preflight import reuse_previous
        reuse_previous(root, durable, args.previous_results.resolve(), args.previous_bundle.resolve())
    elif args.command == 'preflight':
        verify_bundle(root); value = binding(root, durable)
        expected = dict(status='passed', **value)
        if (durable/'preflight.json').exists() and read(durable/'preflight.json') == expected:
            print('Reusing GPU preflight.'); return
        quota(durable)
        for fold in FOLDS:
            command, env = child(root, durable, 'preflight', '--fold', fold)
            subprocess.run(command, env=env, check=True)
        write(durable/'preflight.json', expected)
    elif args.command == 'run': run(root, durable, args.max_runs, args.hours)
    else: report(root, durable)


if __name__ == '__main__': main()
