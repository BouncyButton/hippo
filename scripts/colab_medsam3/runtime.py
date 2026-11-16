"""Recovery, environment binding and calibration receipts for a single adapter run."""
from concurrent.futures import ThreadPoolExecutor

import hashlib

import importlib.metadata

import json

import os

from pathlib import Path

import shutil

import tarfile

import tempfile

import time

ARMS = ('baseline', 'bands', 'sdf', 'bands_sdf', 'bands_edge',
        'bands_edge_sdf', 'depth', 'bands_depth', 'edge_depth', 'sdf_depth',
        'bands_edge_depth', 'bands_sdf_depth', 'edge_sdf_depth', 'bands_edge_sdf_depth')

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

FOLDS, SEEDS = (0, 1, 3), (17, 83, 191)
