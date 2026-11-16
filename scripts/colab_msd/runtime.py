"""Initialization and checkpoint recovery for one training run."""
import base64

from concurrent.futures import ThreadPoolExecutor

import hashlib

import json

import os

from pathlib import Path

import shutil

import tarfile

import tempfile

import time

def read(path):
    return json.loads(Path(path).read_text())

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    tmp.replace(path)

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()

def safe_member(name):
    p = Path(name)
    if p.is_absolute() or '..' in p.parts:
        raise ValueError('Unsafe relative path: ' + name)
    return p

def copy_verified(source, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + '.uploading')
    shutil.copyfile(source, tmp)
    if digest(source) != digest(tmp):
        raise ValueError('Transfer failed verification: ' + str(target))
    tmp.replace(target)

def tensor_digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def model_digest(model):
    h = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        h.update(key.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def canonical_initialization(builder, protocol, architecture, reference):
    """Replay only verified CPU trunc-normal values, retaining all RNG draws and hash gates."""
    import torch
    from monai.networks.layers import weight_init
    expected = protocol['baseline_reference']['initialization_sha256']
    if reference['model_sha256'] != expected:
        raise ValueError('Initialization reference differs from the frozen protocol')
    original = weight_init._no_grad_trunc_normal_
    calls = reference['calls']
    used = []

    def portable(tensor, mean, std, a, b):
        if len(used) >= len(calls):
            raise ValueError('Unexpected truncated-normal initialization call')
        record = calls[len(used)]
        recipe = dict(shape=list(tensor.shape), dtype=str(tensor.dtype), mean=mean, std=std, a=a, b=b)
        if (tensor.device.type != 'cpu' or tensor.dtype != torch.float32
                or recipe != {k: record[k] for k in recipe}
                or tensor_digest(torch.get_rng_state()) != record['rng_before']):
            raise ValueError('Initialization recipe or incoming RNG changed')
        # Execute the unchanged initializer first, including the original random draws.
        value = original(tensor, mean, std, a, b)
        if tensor_digest(torch.get_rng_state()) != record['rng_after']:
            raise ValueError('Initialization RNG consumption changed')
        canonical = torch.frombuffer(bytearray(base64.b64decode(record['data'])),
                                     dtype=torch.float32).reshape(record['shape'])
        maximum = float((value.detach() - canonical).abs().max())
        # Only small numerical deviations are portable; a different distribution is an error.
        torch.testing.assert_close(value, canonical, rtol=2e-6, atol=5e-8)
        with torch.no_grad():
            value.copy_(canonical)
        used.append(dict(max_abs_difference=maximum, shape=record['shape']))
        return value

    weight_init._no_grad_trunc_normal_ = portable
    try:
        model = builder(protocol, architecture)
    finally:
        weight_init._no_grad_trunc_normal_ = original
    if len(used) != len(calls) or tensor_digest(torch.get_rng_state()) != reference['rng_after_model']:
        raise ValueError('Model initialization call sequence or final RNG changed')
    actual = model_digest(model)
    if actual != expected:
        different = [key for key, value in model.state_dict().items()
                     if key not in reference['tensors']
                     or tensor_digest(value) != reference['tensors'][key]['sha256']]
        raise ValueError(f'Initialization differs outside portable tensors: {different[:12]}')
    return model, dict(model_sha256=actual, rng_preserved=True, portable_calls=used)

def install_initialization(root, experiment):
    """Use identical initialization for preflight, benchmark and every matched training run."""
    reference = read(Path(root)/'initialization_reference.json')['models']
    builder = experiment.build_model
    audits = {}

    def build_model(protocol, architecture):
        name = protocol['model_identity']
        seed = str(protocol['baseline_reference']['seed'])
        model, audit = canonical_initialization(builder, protocol, architecture, reference[name][seed])
        audits[f'{name}/seed{seed}'] = audit
        return model

    experiment.build_model = build_model
    return audits

class RecoveryWriter:
    """One copy in flight; immutable checkpoint inode + captured small metadata."""
    def __init__(self, output, durable, interval=180):
        self.output, self.durable = Path(output), Path(durable)
        self.interval, self.last = interval, -float('inf')
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.future = None

    def capture(self, force=False):
        if self.future:
            if force or self.future.done():
                self.future.result()
            else:
                return
        if not force and time.monotonic() - self.last < self.interval:
            return
        checkpoint = self.output/'checkpoint_latest.pt'
        if not checkpoint.exists():
            return
        tmp = Path(tempfile.mkdtemp(prefix='recovery-', dir=self.output.parent))
        os.link(checkpoint, tmp/checkpoint.name)
        for name in ('config.json', 'budget_ledger.json', 'calibration.json', 'initialization.json', 'runtime_recipe.json'):
            file = self.output/name
            if file.exists():
                shutil.copyfile(file, tmp/name)
        self.future = self.pool.submit(self._persist, tmp)
        self.last = time.monotonic()

    def _persist(self, snapshot):
        try:
            self.durable.mkdir(parents=True, exist_ok=True)
            manifest = {p.name: digest(p) for p in snapshot.iterdir()}
            write(snapshot/'snapshot_manifest.json', manifest)
            archive = snapshot.with_suffix('.tar')
            try:
                with tarfile.open(archive, 'w') as tar:
                    for path in sorted(snapshot.iterdir()):
                        tar.add(path, arcname=path.name, recursive=False)
                sha = digest(archive)
                name = 'recovery-' + sha + '.tar'
                copy_verified(archive, self.durable/name)
                previous = read(self.durable/'latest.json') if (self.durable/'latest.json').exists() else None
                write(self.durable/'latest.json', dict(file=name, sha256=sha,
                    charged_attempts=read(snapshot/'budget_ledger.json')['charged_attempts']))
                if previous and previous['file'] != name:
                    old = self.durable/safe_member(previous['file'])
                    if old.exists() and digest(old) == previous['sha256']:
                        old.unlink()
            finally:
                archive.unlink(missing_ok=True)
        finally:
            shutil.rmtree(snapshot)

    def close(self, force=True):
        try:
            if force:
                self.capture(force=True)
            if self.future:
                self.future.result()
        finally:
            self.pool.shutdown(wait=True)
