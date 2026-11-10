"""Resume the frozen campaign from one verified common warmup checkpoint.

This is an external recovery wrapper. It does not edit the frozen training,
loss, reporting code, protocol, tolerances, or accepted completed runs.
"""
import argparse
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def history(folder):
    with (Path(folder)/'metrics.csv').open() as stream:
        return list(csv.DictReader(stream))


def check_prefix(rows, reference, epochs):
    if len(rows) < epochs or len(reference) < epochs:
        raise ValueError('Incomplete warmup')
    deltas = []
    for i, (left, right) in enumerate(zip(rows[:epochs], reference[:epochs]), 1):
        if int(left['epoch']) != i or int(right['epoch']) != i:
            raise ValueError('Wrong epoch order')
        for key in ('constraint_scale', 'train_bands', 'train_edge', 'coefficient_bands', 'coefficient_edge'):
            if float(left[key]) != 0 or float(right[key]) != 0:
                raise ValueError('Constraint active during warmup')
        for key in ('train_dice', 'validation_dice'):
            a, b = float(left[key]), float(right[key])
            if not math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-7):
                raise ValueError(f'Unmatched Dice warmup: epoch {i}, {key}, delta={a-b}')
        deltas.append({k: float(left[k])-float(right[k]) for k in ('train_dice', 'validation_dice')})
    return deltas


def band_reference(root):
    return json.loads((root/'runs/bands/config.json').read_text())['calibration']['coefficients']['bands']


def check_band(value, expected):
    if not math.isclose(value, expected, rel_tol=1e-6, abs_tol=1e-8):
        raise ValueError(f'Unmatched band coefficient: {value} vs {expected}')


def check_completed(p, root, entry):
    from scripts.edge_ablation.train import completed_record, identity
    folder = root/'runs'/entry['id']
    record = completed_record(folder, identity(p, entry))
    if record is not None:
        check_prefix(history(folder), history(root/'runs/bands'), p['calibration']['prefix_epochs'])
        check_band(json.loads((folder/'config.json').read_text())['calibration']['coefficients']['bands'], band_reference(root))
    return record


def prefix_probe(p, root, repair, attempt, device='cuda'):
    import torch
    from scripts.edge_ablation import train
    from scripts.depth_supervision.utils import atomic_json, digest, restore_rng
    entry = next(e for e in p['runs'] if e['id'] == 'bands')
    probe = repair/f'prefix_attempt_{attempt}'
    prefix = p['calibration']['prefix_epochs']
    train.run(p, probe, entry, stop_after=prefix, device_override=device)
    folder = probe/'runs/bands'
    deltas = check_prefix(history(folder), history(root/'runs/bands'), prefix)
    state = torch.load(folder/'latest.pt', map_location='cpu', weights_only=False)
    assert state['identity'] == train.identity(p, entry)
    assert len(state['history']) == prefix and state['calibration'] is None
    model = train.model_factory(torch.device(device))
    model.load_state_dict(state['model'])
    patients = train.Patients(train._load_pkl_dataframe(p['pkl']), p['splits']['5pct']['train'], p['native_geometry'])
    restore_rng(state['rng'])
    calibration = train.calibrate(model, patients, train.Targets(probe, p, entry), torch.device(device), p['calibration'], entry)
    check_band(calibration['coefficients']['bands'], band_reference(root))
    value = dict(folder=str(folder), protocol_identity=p['identity_sha'], prefix_epochs=prefix,
                 prefix_deltas=deltas, calibration=calibration,
                 artifacts={n: digest(folder/n) for n in ('latest.pt', 'best.pt', 'metrics.csv')})
    atomic_json(repair/'verified_prefix.json', value)
    return value


def seed_run(p, root, entry, prefix_record):
    import torch
    from scripts.edge_ablation.train import identity
    from scripts.depth_supervision.utils import atomic_json, digest, save_checkpoint
    source = Path(prefix_record['folder'])
    if prefix_record['protocol_identity'] != p['identity_sha']:
        raise ValueError('Prefix protocol changed')
    for name, expected in prefix_record['artifacts'].items():
        if digest(source/name) != expected:
            raise ValueError('Verified prefix changed: '+name)
    output = root/'runs'/entry['id']
    if output.exists() and any(output.iterdir()):
        raise ValueError('Refuse to overwrite a run: '+entry['id'])
    output.mkdir(parents=True, exist_ok=True)
    for name in ('latest.pt', 'best.pt'):
        state = torch.load(source/name, map_location='cpu', weights_only=False)
        # Only run identity changes. Model, optimizer, scheduler, case order,
        # RNG and epoch history are restored without recomputation.
        state['identity'] = identity(p, entry)
        save_checkpoint(output/name, state)
    atomic_json(output/'shared_warmup.json', prefix_record)


def train_entry(p, root, repair, entry):
    from scripts.edge_ablation import train
    prefix_record = json.loads((repair/'verified_prefix.json').read_text())
    provenance = root/'runs'/entry['id']/'shared_warmup.json'
    if provenance.exists():
        if json.loads(provenance.read_text()) != prefix_record:
            raise ValueError('Resume prefix changed')
    else:
        seed_run(p, root, entry, prefix_record)
    # Catch any calibration mismatch before committing a whole scientific run.
    train.run(p, root, entry, stop_after=p['calibration']['prefix_epochs']+1)
    config = json.loads((root/'runs'/entry['id']/'config.json').read_text())
    check_band(config['calibration']['coefficients']['bands'], band_reference(root))
    train.run(p, root, entry)
    check_completed(p, root, entry)


def worker(p, root, repair):
    import torch
    from scripts.edge_ablation.worker import verify
    from scripts.edge_ablation.report import report
    from scripts.depth_supervision.utils import atomic_json, digest
    (repair/'logs').mkdir(exist_ok=True)
    with (repair/'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name, expected in json.loads((repair/'manifest.json').read_text()).items():
            if digest(repair/name) != expected:
                raise ValueError('Recovery source changed: '+name)
        for name in ('status.json', 'environment.json', 'results.json', 'RESULTS.md'):
            destination = repair/('original_'+name)
            if (root/name).exists() and not destination.exists():
                shutil.copy2(root/name, destination)
        verify(root)
        state = dict(job_id=os.environ['SLURM_JOB_ID'], original_job_id='690510',
                     state='running', current_stage='verify_completed', stages=[], started_unix=time.time(),
                     recovery='Shared warmup; original strict comparisons retained')
        def save():
            atomic_json(repair/'status.json', state)
            atomic_json(root/'status.json', state)
        def execute(stage, mode, extra=()):
            state['current_stage'] = stage
            save()
            start = time.monotonic()
            with (repair/'logs'/(stage+'.log')).open('a', buffering=1) as stream:
                result = subprocess.run([sys.executable, '-u', str(repair/'recover.py'), mode,
                                         '--root', str(root), '--repair', str(repair), *extra],
                                        cwd=root/'source', stdout=stream, stderr=subprocess.STDOUT)
            state['stages'].append(dict(stage=stage, returncode=result.returncode, seconds=time.monotonic()-start))
            save()
            return result.returncode
        try:
            rejected, accepted = [], []
            for entry in p['runs']:
                try:
                    if check_completed(p, root, entry) is not None:
                        accepted.append(entry['id'])
                except ValueError as error:
                    # Hash/identity corruption is never treated as numerical drift.
                    if not str(error).startswith(('Unmatched Dice warmup', 'Unmatched band coefficient')):
                        raise
                    rejected.append(dict(id=entry['id'], reason=str(error)))
            atomic_json(repair/'audit.json', dict(accepted=accepted, rejected=rejected, strict_tolerances_unchanged=True))
            if not (repair/'verified_prefix.json').exists():
                for attempt in range(1, 4):
                    if execute(f'prefix_probe_{attempt}', 'prefix', ('--attempt', str(attempt))) == 0:
                        break
                else:
                    raise RuntimeError('No prefix matched baseline after three probes; preserve evidence and stop')
            # Exclusion is based only on pre-constraint matching, never outcome rank.
            for item in rejected:
                destination = repair/'excluded'/item['id']
                destination.parent.mkdir(exist_ok=True)
                if destination.exists():
                    raise RuntimeError('An excluded trial is already preserved')
                (root/'runs'/item['id']).rename(destination)
            report(p, root)
            for entry in p['runs']:
                if entry['id'] in accepted:
                    continue
                if execute(entry['id'], 'entry', ('--run-id', entry['id'])):
                    raise RuntimeError('Recovered run failed: '+entry['id'])
                result = report(p, root)
                state['completed_runs'] = result['completed']
                save()
            result = report(p, root)
            assert result['completed'] == len(p['runs']) == 20
            state.update(state='complete', current_stage=None, completed_runs=20, finished_unix=time.time())
            save()
        except Exception as error:
            state.update(state='failed', error=repr(error), finished_unix=time.time())
            save()
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('worker', 'prefix', 'entry'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--repair', type=Path, required=True)
    parser.add_argument('--attempt', type=int, default=1)
    parser.add_argument('--run-id')
    args = parser.parse_args()
    sys.path.insert(0, str(args.root/'source'))
    import torch
    torch.set_num_threads(4)
    p = json.loads((args.root/'protocol.json').read_text())
    if args.mode == 'worker':
        worker(p, args.root, args.repair)
    elif args.mode == 'prefix':
        value = prefix_probe(p, args.root, args.repair, args.attempt)
        print(json.dumps(dict(passed=True, folder=value['folder'], prefix_deltas=value['prefix_deltas'])), flush=True)
    else:
        train_entry(p, args.root, args.repair, next(e for e in p['runs'] if e['id'] == args.run_id))


if __name__ == '__main__':
    main()
