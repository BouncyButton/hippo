"""Drive persistence and source binding for the GeoFWI3D SAM3 exploration."""
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
    if not name or path == Path('.') or path.is_absolute() or '..' in path.parts: raise ValueError('Unsafe archive path')
    return path

def copy_verified(source, destination):
    destination = Path(destination); destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(destination.name + '.uploading')
    shutil.copyfile(source, temp)
    if digest(temp) != digest(source): raise IOError('Copy hash mismatch: ' + str(destination))
    temp.replace(destination)

def activate(root):
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root/'source'))
    sys.path.insert(0, str(root/'source/scripts'))

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
        raise ValueError('GPU/software or package changed; use the original environment or a new RESULTS folder')
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

def child(root, durable, command, *extra, deadline=None):
    env = os.environ.copy()
    env.update(PYTHONPATH=str(root/'source'), PYTHONUNBUFFERED='1', WANDB_MODE='disabled',
        MPLCONFIGDIR=str(root/'matplotlib_cache'))
    args = [sys.executable, str(root/'engine.py'), command, '--bundle', str(root), '--durable', str(durable), *map(str, extra)]
    if deadline: args += ['--deadline', str(deadline)]
    return args, env

FOLDS = (0,)


def protocol(root, task):
    p = read(root/'protocols'/task['protocol']/'protocol.json')
    return {**p, 'data': str(root/'data'), 'upstream': str(root/'upstream'),
            'arm': task['scientific_arm']}


def verify_plan(root):
    """Check the original sweep or its explicit 25-update derivative."""
    plan = read(root/'plan.json')
    tasks = plan['tasks']
    reference = read(root/'reference/plan.json')['tasks']
    focused = plan.get('fixed_updates') == 25
    ten_only = plan.get('constraint_only_10pct', False)
    if ten_only and not focused:
        raise ValueError('Constraint-only scope requires 25 updates')
    if 'fixed_updates' in plan and not focused:
        raise ValueError('Unsupported fixed-update exploration')
    if focused:
        reference = [t for t in reference if t['updates'] == 50]
    if ten_only:
        reference = [t for t in reference if (t['arm'], t['bands_target'], t['edge_target'])
                     in [('dice_bands', .1, None), ('dice_bands_edge', .2, .01)]]
    def revised(value):
        value = value.replace('_u50_', '_u25_').replace('u50_', 'u25_', 1)
        return value.replace('b20_e10', 'b10_e100') if ten_only else value
    expected_count = 2 if ten_only else (8 if focused else 24)
    if len(tasks) != expected_count or len({t['id'] for t in tasks}) != expected_count:
        raise ValueError(f'Expected {expected_count} frozen SAM3 conditions')
    expected_ids = {revised(t['id']) if focused else t['id'] for t in reference}
    if {t['id'] for t in tasks} != expected_ids:
        raise ValueError('Condition inventory differs')
    for task in tasks:
        original = next(t for t in reference if (revised(t['id']) if focused else t['id']) == task['id'])
        old = dict(original)
        if focused:
            old.update(id=task['id'], updates=25, protocol=revised(old['protocol']))
        if ten_only and old['arm'] == 'dice_bands_edge':
            old.update(bands_target=.1, edge_target=.1)
        restored = {k:v for k,v in task.items() if k != 'scientific_arm'}
        restored['arm'] = task['scientific_arm']
        if restored != old or task['arm'] != ('baseline' if old['arm']=='dice' else old['arm']):
            raise ValueError('Task differs from the original exploration')
        p = read(root/'protocols'/task['protocol']/'protocol.json')
        if focused:
            expected = read(root/'reference/protocols'/original['protocol']/'protocol.json')
            expected['updates'] = 25
            expected['sampling'] = expected['sampling'].replace('50 updates,', '25 updates,', 1)
            if ten_only and old['arm'] == 'dice_bands_edge':
                expected.update(gradient_ratio_target=.1, gradient_ratio_p95_cap=.5,
                                gradient_ratio_targets=dict(bands=.1, edge=.1),
                                gradient_ratio_p95_caps=dict(bands=.5, edge=.5))
            if p != expected:
                raise ValueError('Focused protocol changed beyond the stopping update')
        if (p['dataset'],p['fold'],p['seed'],p['augmentation']) != ('GeoFWI3D',0,0,'none'):
            raise ValueError('Experiment scope changed')
        if (p['updates'] != task['updates'] or p['learning_rate_horizon_updates'] != 200
                or len(p['train_cases']) != 5 or len(p['evaluation_cases']) != 96
                or set(p['train_cases']) & set(p['evaluation_cases'])):
            raise ValueError('Stopping point, schedule or cohorts changed')
    return plan


def verify_bundle(root):
    for name, expected in read(root/'bundle_manifest.json').items():
        if digest(root/relative(name)) != expected:
            raise ValueError('Changed bundle member: '+name)
    plan = verify_plan(root)
    for name, expected in read(root/'data_manifest.json').items():
        if digest(root/'data'/relative(name)) != expected:
            raise ValueError('Dataset differs: '+name)
    if digest(root/'weights/sam3.pt') != plan['base_sha256']:
        raise ValueError('SAM3 initialization checkpoint differs')
    return plan


def archive_run(root, output, durable, recovery, task):
    """Verify all 101 evaluations before retiring only this run's checkpoints."""
    done = read(output/'completion.json')
    p = read(output/'protocol.json')
    evaluation = read(output/'evaluation.json')
    expected = {(s,c) for s,cases in [('training_support',p['train_cases']),
                                     ('validation',p['evaluation_cases'])] for c in cases}
    records = evaluation['records']
    if (done['status'] != 'complete' or done['updates'] != task['updates']
            or len(records) != len(expected)
            or {(r['split'],r['case_name']) for r in records} != expected
            or any('confusion' not in r or 'foreground_fn' not in r or 'foreground_fp' not in r for r in records)
            or len(read(output/'training_history.json')) != task['updates']):
        raise ValueError('Incomplete evaluation/training cannot be retired')
    summary = read(output/'training_summary.json')
    if summary['frozen_before'] != summary['frozen_after']:
        raise ValueError('Frozen model changed')
    required = ['protocol.json','calibration.json','training_history.json','training_summary.json',
                'training_inventory.json','slab_schedule.json','runtime_recipe.json','evaluation.json',
                'curves.png','completion.json','train.log']
    required += [n for n in ('evaluation_audit.json','evaluation_runtime.json') if (output/n).exists()]
    files = {n:digest(output/n) for n in required}
    target = durable/'runs'/task['id']
    checkpoint = output/'checkpoint_latest.pt'
    if checkpoint.exists() and digest(checkpoint) != done['checkpoint_sha256']:
        raise ValueError('Evaluated checkpoint changed')
    manifest = dict(task=task,files=files,bundle_sha256=digest(root/'bundle_manifest.json'),
                    checkpoint_sha256=done['checkpoint_sha256'])
    if not checkpoint.exists() and (not (target/'archive_manifest.json').exists()
                                   or read(target/'archive_manifest.json') != manifest):
        raise ValueError('Checkpoint missing without a verified archive')
    for name in files:
        copy_verified(output/name,target/name)
    write(target/'archive_manifest.json',manifest)
    for name,sha in files.items():
        if digest(target/name) != sha:
            raise IOError('Archive readback failed')
    removed=[]
    if checkpoint.exists():
        removed.append(dict(file=str(checkpoint),bytes=checkpoint.stat().st_size,sha256=digest(checkpoint)))
        checkpoint.unlink()
    if (recovery/'latest.json').exists():
        m=read(recovery/'latest.json'); path=recovery/relative(m['file'])
        if path.exists():
            if digest(path)!=m['sha256']: raise ValueError('Recovery changed before cleanup')
            removed.append(dict(file=str(path),bytes=path.stat().st_size,sha256=m['sha256']))
            path.unlink()
        (recovery/'latest.json').unlink()
    (recovery/'charged_ledger.json').unlink(missing_ok=True)
    # Partial predictions are only useful for active evaluation of this checkpoint.
    for folder in (output,durable/'parallel_resume'/task['id']):
        for name in ('evaluation_partial.npz','evaluation_partial.json'):
            path=folder/name
            if path.exists():
                removed.append(dict(file=str(path),bytes=path.stat().st_size,sha256=digest(path)))
                path.unlink()
    write(target/'cleanup.json',dict(status='complete',removed=removed,verified_at=time.time()))


def previous_baseline(root, durable, previous):
    """Read an intact completed control; never train or evaluate a replacement."""
    task = next(t for t in read(root/'reference/plan.json')['tasks']
                if t['arm'] == 'dice' and t['updates'] == 25)
    folder = previous/'runs'/task['id']
    archive = read(folder/'archive_manifest.json')
    expected_task = dict(task, arm='baseline', scientific_arm='dice')
    if archive['task'] != expected_task or read(folder/'cleanup.json')['status'] != 'complete':
        raise ValueError('Previous baseline is not a completed 25-update control')
    parent_sha = read(root/'plan.json')['parent_colab_manifest_sha256']
    old_binding = read(previous/'environment_binding.json')
    if archive['bundle_sha256'] != parent_sha or old_binding['bundle_sha256'] != parent_sha:
        raise ValueError('Previous baseline source differs')
    if old_binding['environment'] != read(durable/'environment_binding.json')['environment']:
        raise ValueError('Previous baseline GPU/software differs; cannot call this a matched comparison')
    for name, sha in archive['files'].items():
        if digest(folder/relative(name)) != sha:
            raise ValueError('Previous baseline archive changed: '+name)
    expected = read(root/'reference/protocols'/task['protocol']/'protocol.json')
    actual = read(folder/'protocol.json')
    actual = {k:v for k,v in actual.items() if k not in ('data','upstream','arm')}
    expected = {k:v for k,v in expected.items() if k not in ('data','upstream','arm')}
    if actual != expected:
        raise ValueError('Previous baseline protocol differs')
    done = read(folder/'completion.json')
    if done['status'] != 'complete' or done['updates'] != 25:
        raise ValueError('Previous baseline is incomplete')
    records = read(folder/'evaluation.json')['records']
    cohort = {(s,c) for s,key in [('training_support','train_cases'),('validation','evaluation_cases')]
              for c in expected[key]}
    if len(records) != 101 or {(r['split'],r['case_name']) for r in records} != cohort:
        raise ValueError('Previous baseline cohort differs')
    return {**expected_task, **done, 'reused_from':str(folder)}, records


def report(root,durable,previous=None):
    rows=[]; cases={}
    plan = read(root/'plan.json')
    for task in plan['tasks']:
        if completed(root,durable,task):
            output=durable/'runs'/task['id']; done=read(output/'completion.json')
            records=read(output/'evaluation.json')['records']
            cases[task['id']]={r['case_name']:r['macro_hard_dice'] for r in records if r['split']=='validation'}
            rows.append({**task,**done})
    native={r['updates']:r for r in rows if r['arm']=='baseline'}
    reused = None
    if previous is not None:
        reused, records = previous_baseline(root, durable, previous)
        native[25] = reused
        cases[reused['id']] = {r['case_name']:r['macro_hard_dice'] for r in records if r['split']=='validation'}
        truths = {r['case_name']:r['truth_sha256'] for r in records if r['split']=='validation'}
        for row in rows:
            new_records = read(durable/'runs'/row['id']/'evaluation.json')['records']
            if {r['case_name']:r['truth_sha256'] for r in new_records if r['split']=='validation'} != truths:
                raise ValueError('Previous baseline validation annotations differ')
    bands={r['updates']:r for r in rows if r['scientific_arm']=='dice_bands' and r['bands_target']==.2}
    bands10={r['updates']:r for r in rows if r['scientific_arm']=='dice_bands' and r['bands_target']==.1}
    import numpy as np
    for row in rows:
        for label,lookup in [('native',native),('bands20',bands),('bands10',bands10)]:
            control=lookup.get(row['updates'])
            row['gain_vs_'+label+'_pp']=None
            if control:
                a,b=cases[row['id']],cases[control['id']]
                if a.keys()!=b.keys(): raise ValueError('Paired validation cohort differs')
                delta=np.array([100*(a[k]-b[k]) for k in sorted(a)])
                row['gain_vs_'+label+'_pp']=float(delta.mean())
                row['paired_vs_'+label]=dict(wins=int((delta>0).sum()),losses=int((delta<0).sum()),
                    ties=int((delta==0).sum()),p10_pp=float(np.quantile(delta,.1)),
                    worst_pp=float(delta.min()),case_delta_pp=dict(zip(sorted(a),delta.tolist())))
    value=dict(new_completed=len(rows),new_total=len(plan['tasks']),runs=rows,
        reused_baseline=reused,
        historical_reference=read(root/'reference/comparison.json'),
        interpretation='Fresh Colab controls at matching update counts; cluster results are reference only. '
        'Five training volumes; all96 validation volumes; fold0/seed0; exploratory; no augmentation; update matched, not FLOP capped.')
    if reused:
        value['interpretation'] = 'Existing completed 25-update Colab baseline reused after archive, source, protocol, cohort and environment checks. No baseline training or evaluation was repeated.'
    write(durable/'comparison.json',value)
    print(f"Archived {len(rows)}/{len(plan['tasks'])} runs",flush=True)
    return value


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=('verify','preflight','report'))
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--durable',type=Path,required=True)
    parser.add_argument('--baseline-results',type=Path)
    args=parser.parse_args(); root=args.bundle.resolve(); durable=args.durable.resolve()
    durable.mkdir(parents=True,exist_ok=True); activate(root)
    if args.command=='report': report(root,durable,args.baseline_results); return
    plan = verify_bundle(root)
    if args.command=='verify': print(f"Verified {len(plan['tasks'])} GeoFWI3D SAM3 conditions and all101 volumes"); return
    from parallel import SweepLock
    with SweepLock(root):
        value=binding(root,durable); expected=dict(status='passed',**value)
        if (durable/'preflight.json').exists() and read(durable/'preflight.json')==expected:
            fold_calibration(root,durable,0)
            print('Reusing matching GPU preflight and shared calibration'); return
        quota(durable)
        command,env=child(root,durable,'preflight')
        subprocess.run(command,env=env,check=True)
        write(durable/'preflight.json',expected)


if __name__=='__main__': main()
