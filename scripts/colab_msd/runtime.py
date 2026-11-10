"""Colab orchestration; never submits to or changes the cluster.

Training/data stay on the VM SSD. A single asynchronous writer persists coherent
recovery snapshots; the small charge ledger is written before each update.
Unexpected lost updates are detected, never silently removed from the budget.
"""
import argparse
import base64
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


def verify_bundle(root):
    root = Path(root)
    for name, sha in read(root/'bundle_manifest.json').items():
        if digest(root/safe_member(name)) != sha:
            raise ValueError('Bundle changed: ' + name)
    plan = read(root/'plan.json')
    if plan['augmentation'] != 'none' or len(plan['tasks']) != 504:
        raise ValueError('Expected exactly 504 unaugmented MSD experiments')
    identities = set()
    for task in plan['tasks']:
        identity = (task['model'], task['arm'], task['fold'], task['seed'])
        if identity in identities:
            raise ValueError('Duplicate experiment')
        identities.add(identity)
        p = read(root/'protocols'/task['protocol']/'protocol.json')
        native = task['model'] == 'nnunet'
        if (p['augmentation'] != 'none' or p['folds'] != [0, 1, 2]
                or p['seeds'] != [task['seed']] or p['weight_ce'] != int(native)
                or p['calibration_target_ratio'] != (.6 if native else .1)
                or p['budget_flops'] != 31976450667090):
            raise ValueError('Scientific settings changed')
        splits = read(root/'protocols'/task['protocol']/'splits_10patients.json')
        for split in splits:
            if (len(set(split['train'])) != 10 or len(set(split['val'])) != 52
                    or set(split['train']) & set(split['val'])):
                raise ValueError('Invalid frozen MSD split')
    return plan


def activate(root):
    root = Path(root).resolve()
    for path in reversed([root/'source', root/'source/scripts', root/'source/baselines/nnUNet']):
        sys.path.insert(0, str(path))
    return root


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


def environment():
    import torch
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:
        raise RuntimeError('Native BF16 GPU required (for example H100, A100, L4 or RTX PRO 6000 Blackwell). T4 cannot preserve this precision recipe.')
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 is unavailable; precision will not be changed automatically')
    packages = ('torch', 'monai', 'dynamic-network-architectures', 'numpy', 'pandas',
                'scipy', 'scikit-image', 'nibabel', 'einops')
    result = dict(packages={p: importlib.metadata.version(p) for p in packages},
                  gpu=torch.cuda.get_device_name(), cuda=torch.version.cuda,
                  cudnn=torch.backends.cudnn.version())
    if result['packages']['monai'] != '1.6.0' or result['packages']['dynamic-network-architectures'] != '0.4.4':
        raise ValueError('Install the pinned model dependencies first')
    return result


def binding(root, durable):
    value = dict(bundle_sha256=digest(root/'bundle_manifest.json'), environment=environment())
    path = durable/'environment_binding.json'
    if path.exists() and read(path) != value:
        has_work = any((durable/'runs').glob('*/archive_manifest.json')) or any((durable/'recovery').glob('*/charged_ledger.json'))
        if has_work:
            raise ValueError('GPU/software changed within the replication. Restore the original runtime; do not mix environments.')
    write(path, value)
    return value


def preflight(root, durable):
    """Check all initializations and real CUDA equivalence before any study run."""
    plan = verify_bundle(root)
    value = binding(root, durable)
    receipt = durable/'preflight.json'
    if receipt.exists() and read(receipt) == dict(status='passed', **value):
        print('Reusing verified preflight for the same bundle, GPU and software.', flush=True)
        return
    import gc
    from scripts.cross_dataset_campaign import conventional as exp
    audits = install_initialization(root, exp)
    for name in plan['models']:
        for seed in (0, 1, 2):
            folder = root/'protocols'/f'{name}_s{seed}'
            p = read(folder/'protocol.json')
            exp.base.seed_everything(seed)
            model = exp.build_model(p, read(folder/'nnunet_architecture.json'))
            h = hashlib.sha256()
            for key, tensor in sorted(model.state_dict().items()):
                h.update(key.encode()); h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
            if h.hexdigest() != p['baseline_reference']['initialization_sha256']:
                raise ValueError(f'Initialization differs for {name}/seed{seed}')
            del model; gc.collect()
            print(f'Exact original initialization verified: {name}/seed{seed}', flush=True)
    write(durable/'initialization_portability.json', audits)
    # Run the already audited reference/cached numerical benchmark on this GPU.
    # Its original batch-size trials are diagnostics only; training stays batch 1.
    scratch = root/'benchmark_stage'
    scratch.mkdir(exist_ok=True)
    for name in plan['models']:
        target = scratch/'protocols'/name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(root/'protocols'/f'{name}_s0', target)
        p = read(target/'protocol.json')
        p['data'] = str(root/'data/msd_hippocampus_full.pkl')
        write(target/'protocol.json', p)
    out = root/'benchmark_outputs'
    command = [sys.executable, str(root/'entry.py'), '--benchmark',
               '--stage', str(scratch), '--output', str(out)]
    subprocess.run(command, cwd=root/'source', env=child_environment(root), check=True)
    if read(out/'summary.json')['status'] != 'passed':
        raise ValueError('GPU equivalence benchmark failed')
    for file in out.glob('*.json'):
        copy_verified(file, durable/'preflight'/file.name)
    # Retire only generated benchmark checkpoints after its reports are verified.
    for file in out.rglob('*.pt'):
        file.unlink()
    write(receipt, dict(status='passed', **value))


def child_environment(root):
    env = dict(os.environ)
    env['PYTHONPATH'] = ':'.join(str(root/p) for p in ('source', 'source/scripts', 'source/baselines/nnUNet'))
    env['MPLCONFIGDIR'] = str(root/'matplotlib_cache')
    env['WANDB_MODE'] = 'disabled'
    env['PYTHONUNBUFFERED'] = '1'
    return env


def check_drive_headroom(durable):
    """Read server quota, including trash, before consuming another run's budget."""
    import google.auth
    from googleapiclient.discovery import build
    credentials, _ = google.auth.default()
    quota = build('drive', 'v3', credentials=credentials, cache_discovery=False).about().get(
        fields='storageQuota').execute()['storageQuota']
    write(durable/'storage_quota.json', dict(checked_at=time.time(), **quota))
    if 'limit' in quota and int(quota['limit']) - int(quota['usage']) < 8 * 1024**3:
        raise RuntimeError('Drive headroom below 8 GiB; current recovery is preserved. Check quota and Drive trash before continuing.')


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


def restore_recovery(output, durable):
    marker = durable/'latest.json'
    if not marker.exists():
        if (durable/'charged_ledger.json').exists():
            raise RuntimeError('Charged work exists without a durable checkpoint. Retained for audit; cannot silently restart.')
        return
    m = read(marker)
    archive = durable/safe_member(m['file'])
    if digest(archive) != m['sha256']:
        raise ValueError('Recovery archive hash differs')
    output.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            safe_member(member.name)
            if not member.isfile():
                raise ValueError('Recovery must contain regular files only')
        tar.extractall(output, filter='data')
    for name, sha in read(output/'snapshot_manifest.json').items():
        if digest(output/safe_member(name)) != sha:
            raise ValueError('Recovery member hash differs')
    (output/'snapshot_manifest.json').unlink()
    if (durable/'charged_ledger.json').exists():
        latest = read(durable/'charged_ledger.json')
        saved = read(output/'budget_ledger.json')
        if latest != saved:
            raise RuntimeError('Abrupt interruption left charged work newer than the durable checkpoint. Recovery is preserved; budget audit required before resuming.')


def archive_completed(root, output, durable, recovery, task):
    from scripts.cross_dataset_campaign import storage
    manifest = storage.prepare(output, task)
    target = durable/'runs'/task['id']
    for name in [*manifest['metadata'], 'archive_manifest.json']:
        copy_verified(output/name, target/name)
    storage.verify_archive(target, manifest)
    ack = dict(status='verified', manifest_sha256=digest(target/'archive_manifest.json'),
               source_verified=True, source_manifest_sha256=digest(root/'bundle_manifest.json'),
               durable_archive=str(target), verified_at=time.time())
    write(target/'archive_acknowledgment.json', ack)
    write(output/'archive_acknowledgment.json', ack)
    receipt = storage.retire(output)
    write(target/'cleanup.json', receipt)
    # Delete only this completed cell's verified recovery checkpoint.
    marker = recovery/'latest.json'
    if marker.exists():
        r = read(marker); path = recovery/safe_member(r['file'])
        if path.exists() and digest(path) == r['sha256']:
            path.unlink(); marker.unlink()
    (recovery/'charged_ledger.json').unlink(missing_ok=True)


def completed(durable, task):
    from scripts.cross_dataset_campaign.storage import verify_archive
    out = durable/'runs'/task['id']
    if not (out/'cleanup.json').exists():
        return False
    if read(out/'cleanup.json')['status'] != 'complete':
        return False
    manifest = read(out/'archive_manifest.json')
    ack = read(out/'archive_acknowledgment.json')
    if (manifest['task'] != task or ack['manifest_sha256'] != digest(out/'archive_manifest.json')
            or ack['status'] != 'verified' or not ack['source_verified']
            or ack['source_manifest_sha256'] != read(durable/'environment_binding.json')['bundle_sha256']):
        raise ValueError('Completed cell identity or archive acknowledgment changed')
    verify_archive(out, manifest)
    return True


def run_sweep(root, durable, max_runs=504, hours=10):
    plan = verify_bundle(root)
    value = binding(root, durable)
    if read(durable/'preflight.json') != dict(status='passed', **value):
        raise RuntimeError('Run the GPU preflight cell first')
    deadline = time.time() + hours * 3600
    done_now = 0
    work = root/'work'; work.mkdir(exist_ok=True)
    with (work/'sweep.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows, _ = report(root, durable)
        verified_ids = {row['id'] for row in rows}
        for task in plan['tasks']:
            if task['id'] in verified_ids:
                continue
            if done_now >= max_runs or time.time() > deadline - 120:
                break
            if (durable/'STOP').exists():
                print('STOP requested; remove the STOP file to resume.'); break
            output = work/task['id']; recovery = durable/'recovery'/task['id']
            if not output.exists() or not any(output.iterdir()):
                restore_recovery(output, recovery)
            output.mkdir(exist_ok=True)
            if shutil.disk_usage(root).free < 12 * 1024**3:
                raise RuntimeError('Keep at least 12 GiB free on the runtime SSD')
            check_drive_headroom(durable)
            if not (output/'completion.json').exists():
                p = read(root/'protocols'/task['protocol']/'protocol.json')
                command = [sys.executable, str(root/'entry.py'), '--model', p['trainer_alias'],
                    '--objective', task['arm'], '--fold', str(task['fold']), '--seed', str(task['seed']),
                    '--pkl', str(root/'data/msd_hippocampus_full.pkl'), '--output', str(output),
                    '--protocol-dir', str(root/'protocols'/task['protocol']), '--resume', '--threads', '2']
                env = child_environment(root)
                env.update(HIPPO_COLAB_RECOVERY=str(recovery), HIPPO_DEADLINE=str(deadline - 60),
                           HIPPO_COLAB_BUNDLE=str(root))
                print('RUN', task['id'], flush=True)
                with (output/'train.log').open('a') as log:
                    proc = subprocess.Popen(command, cwd=root/'source', env=env, stdout=log, stderr=subprocess.STDOUT)
                    sent = False
                    try:
                        while proc.poll() is None:
                            if (durable/'STOP').exists() and not sent:
                                proc.send_signal(signal.SIGUSR1); sent = True
                            time.sleep(5)
                    except KeyboardInterrupt:
                        proc.send_signal(signal.SIGUSR1)
                        print('Saving a coherent recovery checkpoint; waiting for its Drive verification.', flush=True)
                        proc.wait()
                        raise
                if proc.returncode == 75:
                    print('Gracefully saved; rerun this cell to continue.'); break
                if proc.returncode:
                    copy_verified(output/'train.log', durable/'failures'/task['id']/'train.log')
                    raise RuntimeError('Trainer failed; recovery preserved. ' + (output/'train.log').read_text()[-2500:])
            archive_completed(root, output, durable, recovery, task)
            done_now += 1
            print('ARCHIVED', task['id'], flush=True)
            rows.append(result_row(durable, task))
            report(root, durable, rows=rows)
    report(root, durable, rows=rows)


def result_row(durable, task):
    out = durable/'runs'/task['id']; done = read(out/'completion.json')
    e = read(out/'validation_latest.json')
    return dict(**task, dice=100*e['summary']['macro_hard_dice'],
        training_dice=100*done['final_training_hard_dice'], updates=done['updates'],
        training_flops=done['consumed_training_flops'], calibration_flops=done['calibration_flops'],
        fn=sum(c['foreground_fn'] for c in e['cases']), fp=sum(c['foreground_fp'] for c in e['cases']),
        ap_swaps=sum(sum(c['ap_swaps'].values()) for c in e['cases']))


def report(root, durable, rows=None):
    if rows is None:
        rows = [result_row(durable, task) for task in read(root/'plan.json')['tasks'] if completed(durable, task)]
    baselines = {(x['model'], x['fold'], x['seed']):x['dice'] for x in rows if x['arm']=='dice'}
    for x in rows:
        b = baselines.get((x['model'], x['fold'], x['seed']))
        x['gain_pp'] = None if b is None else x['dice'] - b
    aggregate = []
    for model in read(root/'plan.json')['models']:
        for arm in read(root/'plan.json')['arms']:
            r = [x for x in rows if x['model']==model and x['arm']==arm]
            if not r:
                continue
            # Average seeds within each fold first. Folds, not nine runs, are the units.
            folds = [statistics.mean(x['dice'] for x in r if x['fold']==f)
                     for f in (0, 1, 2) if len([x for x in r if x['fold']==f]) == 3]
            aggregate.append(dict(model=model, arm=arm, completed=len(r), expected=9,
                complete_fold_means=folds,
                mean_dice=statistics.mean(folds) if len(folds)==3 else None,
                sd_across_folds=statistics.stdev(folds) if len(folds)==3 else None))
    write(durable/'comparison.json', dict(completed=len(rows), total=504, runs=rows, aggregate=aggregate,
        note='Means across three fold means, each averaging three seeds; overlapping validation folds are not independent test datasets.'))
    print(f'Verified results: {len(rows)}/504', flush=True)
    return rows, aggregate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('verify', 'preflight', 'run', 'report'))
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--durable', type=Path, required=True)
    parser.add_argument('--max-runs', type=int, default=504)
    parser.add_argument('--hours', type=float, default=10)
    args = parser.parse_args()
    root = activate(args.bundle); durable = args.durable.resolve(); durable.mkdir(parents=True, exist_ok=True)
    if args.command == 'verify':
        print(len(verify_bundle(root)['tasks']), 'unaugmented tasks verified')
    elif args.command == 'preflight':
        preflight(root, durable)
    elif args.command == 'run':
        run_sweep(root, durable, args.max_runs, args.hours)
    else:
        report(root, durable)


if __name__ == '__main__':
    main()
