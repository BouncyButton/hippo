"""Run the frozen replication plan sequentially and retain verified metadata."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

CAMPAIGN = 'capped_replication_31976_20261007'
BUDGET = 31976450667090
STOP = False


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        stream.write(json.dumps(value, indent=2) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def check_frozen(stage, pkl=None):
    source = stage / 'source'
    manifest = read(stage / 'source_manifest.json')
    for name, expected in manifest.items():
        if digest(source / name) != expected:
            raise ValueError('Frozen source changed: ' + name)
    directory = source / 'experiments' / CAMPAIGN
    plan = read(directory / 'plan.json')
    if (plan['total_new_runs'] != 448 or plan['budget_flops'] != BUDGET
            or plan['folds'] != [0, 1, 2] or plan['seeds'] != [0, 1, 2]
            or plan['calibration_in_cap'] or plan['checkpoint_retention'] != 'verified_metadata_only'):
        raise ValueError('Authorized replication scope differs')
    identities = {(r['model'], r['objective'], r['fold'], r['seed']) for r in plan['runs']}
    reused = {(r['model'], r['objective'], r['fold'], r['seed']) for r in plan['reused_runs']}
    expected = {(m, arm, f, s) for m in plan['models'] for arm in plan['objectives']
                for f in plan['folds'] for s in plan['seeds']}
    if len(identities) != 448 or len(reused) != 56 or identities & reused or identities | reused != expected:
        raise ValueError('Replication coverage differs')
    if pkl is not None and digest(pkl) != plan['dataset_sha256']:
        raise ValueError('Dataset differs')
    from scripts.capped_depth_delay.schedule import training_flops
    for model in plan['models']:
        protocol_dir = directory / model
        protocol = read(protocol_dir / 'protocol.json')
        for name, expected_sha in protocol['training_source_sha256'].items():
            if digest(source / name) != expected_sha:
                raise ValueError('Protocol source differs: ' + name)
        for name, key in [('splits_10patients.json', 'splits_sha256'),
                          ('nnunet_architecture.json', 'nnunet_architecture_sha256'),
                          ('native_geometry.json', 'native_geometry_sha256')]:
            if digest(protocol_dir / name) != protocol[key]:
                raise ValueError('Frozen input differs: ' + name)
        splits = read(protocol_dir / 'splits_10patients.json')
        for fold in plan['folds']:
            split = splits[fold]
            if (len(set(split['train'])) != 10 or len(set(split['val'])) != 52
                    or set(split['train']) & set(split['val'])
                    or not set(split['train'] + split['val']) <= set(protocol['native_geometry'])):
                raise ValueError('Split or annotation geometry differs')
        for arm in plan['objectives']:
            schedule = protocol['models'][protocol['trainer_alias']]['objectives'][arm]
            n = schedule['max_steps']
            if not training_flops(n, schedule) <= BUDGET < training_flops(n + 1, schedule):
                raise ValueError('Strict budget differs')
            if 'sdf' in arm and schedule['sdf_prefix_updates'] + schedule['sdf_warmup_updates'] > n:
                raise ValueError('SDF ramp does not fit')
    return plan


def verify_stored(output):
    manifest = read(output / 'metadata_manifest.json')
    receipt = read(output / 'storage_acknowledgment.json')
    if not manifest['full_checkpoints_verified_before_discard']:
        raise ValueError('Full checkpoint verification missing')
    if (receipt['manifest_sha256'] != digest(output / 'metadata_manifest.json')
            or not receipt['all_metadata_sha256_verified']):
        raise ValueError('Storage acknowledgment differs')
    for name, expected in manifest['files'].items():
        if Path(name).name != name or digest(output / name) != expected:
            raise ValueError('Stored metadata changed: ' + name)
    return read(output / 'condition_summary.json')


def verify_checkpoints(output, row):
    import torch
    config = read(output / 'config.json')
    completion = read(output / 'completion.json')
    proofs = {}
    for name in ('checkpoint_best.pt', 'checkpoint_latest.pt'):
        payload = torch.load(output / name, map_location='cpu', weights_only=False)
        expected_step = completion['selected_step'] if name == 'checkpoint_best.pt' else row['updates']
        if payload['configuration'] != config or payload['step'] != expected_step:
            raise ValueError('Checkpoint configuration or update differs')
        if name == 'checkpoint_best.pt' and payload['score'] != row['best_dice']:
            raise ValueError('Best checkpoint score differs')
        if name == 'checkpoint_latest.pt' and payload['charged_attempts'] != row['updates']:
            raise ValueError('Final checkpoint compute differs')
        h = hashlib.sha256()
        for key, value in payload['model'].items():
            if not torch.isfinite(value).all():
                raise ValueError('Nonfinite model tensor: ' + key)
            h.update(key.encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
        proofs[name] = dict(file_sha256=digest(output/name), model_tensor_sha256=h.hexdigest(),
                           configuration_verified=True, update=expected_step, finite_tensors=True)
        del payload
    return proofs


def store_metadata(output, destination, row, checkpoint_proofs):
    if set(checkpoint_proofs) != {'checkpoint_best.pt', 'checkpoint_latest.pt'}:
        raise ValueError('Both verified checkpoints are required before metadata storage')
    for name, proof in checkpoint_proofs.items():
        if (proof['file_sha256'] != digest(output/name)
                or not proof['configuration_verified'] or not proof['finite_tensors']):
            raise ValueError('Checkpoint verification proof differs')
    temporary = destination.with_name('run.pending')
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    for path in output.iterdir():
        if path.is_file() and path.suffix in ('.json', '.csv'):
            shutil.copyfile(path, temporary / path.name)
    write(temporary / 'condition_summary.json', row)
    manifest = dict(files={p.name: digest(p) for p in temporary.iterdir() if p.is_file()},
                    full_checkpoints_verified_before_discard=True,
                    discarded_checkpoint_proofs=checkpoint_proofs,
                    retention='JSON/CSV metadata only; full checkpoints were verified before discard')
    write(temporary / 'metadata_manifest.json', manifest)
    write(temporary / 'storage_acknowledgment.json', dict(
        manifest_sha256=digest(temporary / 'metadata_manifest.json'), all_metadata_sha256_verified=True))
    verify_stored(temporary)
    temporary.replace(destination)
    return verify_stored(destination)


def report(root, plan, rows, status):
    all_rows = plan['reused_runs'] + rows
    baselines = {(r['model'], r['fold'], r['seed']): r['final_dice']
                 for r in all_rows if r['objective'] == 'dice'}
    for row in all_rows:
        row['gain_vs_paired_baseline_pp'] = 100 * (row['final_dice'] - baselines[(row['model'], row['fold'], row['seed'])])
    write(root / 'summary.json', dict(status=status, budget_flops=BUDGET, budget_scope='optimization_only',
          calibration_in_cap=False, total_new_runs=448, completed_new_runs=len(rows),
          reused_conditions=56, failures=[], runs=rows, reused_runs=plan['reused_runs']))
    fields = ['model', 'objective', 'fold', 'seed', 'updates', 'epochs', 'final_dice', 'best_dice',
              'anterior_dice', 'posterior_dice', 'gain_vs_paired_baseline_pp',
              'optimization_gflops', 'calibration_gflops']
    temporary = root / 'results.csv.tmp'
    with temporary.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(all_rows)
    temporary.replace(root / 'results.csv')
    lines = [f'Replication status: {status}; {len(rows)}/448 new runs complete; 56 pilot conditions retained.',
             '', 'Final foreground validation Dice. Optimization cap: 31,976.45066709 GFLOPs.',
             'Calibration is reported separately. SDF timing was frozen from fold 0 / seed 0 pilot results.',
             'Each completed run retains its configuration, calibration, case metrics and verified compute ledger.',
             'Model checkpoints are verified before discard and cannot be used for later inference.', '',
             '| Model | Objective | Fold | Seed | Dice % | Gain pp | Updates | Epochs |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in all_rows:
        lines.append('| {model} | {objective} | {fold} | {seed} | {dice:.2f} | {gain:+.2f} | {updates} | {epochs:.1f} |'.format(
            **row, dice=100 * row['final_dice'], gain=row['gain_vs_paired_baseline_pp']))
    temporary = root / 'results.md.tmp'
    temporary.write_text('\n'.join(lines) + '\n')
    temporary.replace(root / 'results.md')


def request_stop(signum, frame):
    global STOP
    STOP = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--pkl', type=Path)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--safe-runtime-seconds', type=int, default=23 * 3600 + 40 * 60)
    args = parser.parse_args()
    plan = check_frozen(args.stage, args.pkl)
    if args.check_only:
        print(json.dumps({'status': 'passed', 'new_runs': 448, 'reused': 56, 'strict_cap_verified': True}))
        return
    if args.root is None or args.pkl is None or os.getuid() != 1395:
        raise ValueError('Cluster account, data and output root required')
    import torch
    import monai
    if torch.__version__ != plan['runtime_versions']['torch'] or monai.__version__ != plan['runtime_versions']['monai']:
        raise ValueError('Use the already validated production runtime')
    import fcntl
    from scripts.capped_replication_verify import verify_run
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source = args.stage / 'source'
        if not (root / 'source').exists():
            (root / 'source').symlink_to(source.resolve(), target_is_directory=True)
        if (root / 'source').resolve() != source.resolve():
            raise ValueError('Resume uses another frozen source')
        plan_sha = digest(source / 'experiments' / CAMPAIGN / 'plan.json')
        binding = dict(plan_sha256=plan_sha, source_manifest_sha256=digest(args.stage / 'source_manifest.json'))
        binding_path = root / 'campaign_binding.json'
        if binding_path.exists() and read(binding_path) != binding:
            raise ValueError('Resume campaign changed')
        write(binding_path, binding)
        signal.signal(signal.SIGUSR1, request_stop)
        start = time.monotonic()
        scratch = Path('/tmp') / f'hippo-replication-{os.getuid()}-{os.environ["SLURM_JOB_ID"]}'
        scratch.mkdir(mode=0o700, exist_ok=True)
        (scratch / 'source').symlink_to(source.resolve(), target_is_directory=True)
        rows = []
        for entry in plan['runs']:
            model, arm, fold, seed = [entry[k] for k in ('model', 'objective', 'fold', 'seed')]
            relative = Path(model) / arm / f'fold{fold}/seed{seed}/run'
            destination = root / relative
            if destination.exists():
                row = verify_stored(destination)
                if (row['model'], row['objective'], row['fold'], row['seed']) != (model, arm, fold, seed):
                    raise ValueError('Stored condition identity differs')
                rows.append(row)
        report(root, plan, rows, 'running')
        complete = {(r['model'], r['objective'], r['fold'], r['seed']) for r in rows}
        for index, entry in enumerate(plan['runs'], 1):
            model, arm, fold, seed = [entry[k] for k in ('model', 'objective', 'fold', 'seed')]
            relative = Path(model) / arm / f'fold{fold}/seed{seed}/run'
            destination = root / relative
            if (model, arm, fold, seed) in complete:
                continue
            if STOP or time.monotonic() - start >= args.safe_runtime_seconds:
                report(root, plan, rows, 'paused_walltime')
                write(root / 'progress.json', dict(status='paused_walltime', completed=len(rows), total=448))
                print(json.dumps({'status': 'paused_walltime', 'completed': len(rows), 'remaining': 448-len(rows)}), flush=True)
                return
            quota = subprocess.check_output(['beegfs', 'quota', 'list-usage', '--uids', 'current', '--pool', 'pool_home'], text=True)
            match = re.search(r'(\d+(?:\.\d+)?)GiB/(\d+(?:\.\d+)?)GiB', quota)
            if not match or float(match[2]) - float(match[1]) < 1:
                report(root, plan, rows, 'paused_storage')
                write(root / 'progress.json', dict(status='paused_storage', completed=len(rows), total=448))
                write(root / 'required_action.json', dict(reason='Need at least 1 GiB quota headroom', quota=quota))
                return
            if shutil.disk_usage(scratch).free < 12 * 1024**3:
                raise RuntimeError('Need 12 GiB free node scratch for transient full checkpoints')
            output = scratch / model / arm / f'fold{fold}/seed{seed}/run'
            directory = source / 'experiments' / CAMPAIGN / model
            write(root / 'progress.json', dict(status='training', completed=len(rows), total=448,
                  current_index=index, model=model, objective=arm, fold=fold, seed=seed))
            try:
                subprocess.run([sys.executable, '-m', 'scripts.capped_replication_train',
                    '--model', entry['trainer_alias'], '--objective', arm, '--fold', str(fold), '--seed', str(seed),
                    '--protocol-dir', str(directory), '--output', str(output), '--pkl', str(args.pkl),
                    '--threads', '4'], check=True)
                row = verify_run(scratch, model, arm, directory, fold=fold, seed=seed)
                for filename in ('validation_best.json', 'validation_latest.json'):
                    val = read(output / filename)
                    anterior = sum(c['anterior_hard_dice'] for c in val['cases']) / 52
                    posterior = sum(c['posterior_hard_dice'] for c in val['cases']) / 52
                    if len(val['cases']) != 52 or abs((anterior + posterior)/2 - val['summary']['macro_hard_dice']) > 1e-12:
                        raise ValueError('Case-level validation average differs')
                row.update(sdf_policy=entry['sdf_policy'], storage_policy='verified_metadata_only')
                destination.parent.mkdir(parents=True, exist_ok=True)
                checkpoint_proofs = verify_checkpoints(output, row)
                row = store_metadata(output, destination, row, checkpoint_proofs)
                rows.append(row)
                report(root, plan, rows, 'running')
                shutil.rmtree(output.parent)
                print(json.dumps({'condition_complete': row, 'completed': len(rows), 'total': 448}), flush=True)
            except BaseException as error:
                report(root, plan, rows, 'failed')
                failure = dict(model=model, objective=arm, fold=fold, seed=seed, error=repr(error))
                write(root / 'failure.json', failure)
                summary = read(root / 'summary.json'); summary['failures'] = [failure]
                write(root / 'summary.json', summary)
                raise
        report(root, plan, rows, 'complete')
        write(root / 'progress.json', dict(status='complete', completed=448, total=448, reused=56))
        print(json.dumps({'campaign_complete': 448, 'reused': 56}), flush=True)


if __name__ == '__main__':
    main()
