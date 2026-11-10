"""Run the fixed 15-cell comparison and retire verified scratch artifacts."""
import argparse
import csv
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time

import numpy as np
import torch
import monai

from scripts.band_focus import combo_train as train
from scripts.band_focus.ap82_shared_tissue import paired_counts, aggregate
from scripts.depth_supervision.utils import atomic_json, digest
from . import objectives

STOP = False
ORIGINAL_EVALUATE = train.evaluate
ORIGINAL_FACTORY = train.model_factory
ORIGINAL_SAVE = train.save_checkpoint


def read(path):
    return json.loads(Path(path).read_text())


def state_digest(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def install(root, entry):
    """Use the established training loop, extending only losses and diagnostics."""
    dest = root/'runs'/entry['id']
    references = {}
    for name in ('dice', 'winning_constraint'):
        path = root/'references'/f'{name}.npz'
        if path.exists() and name != entry['id']:
            with np.load(path, allow_pickle=False) as archive:
                references[name] = {key: archive[key] for key in archive.files}

    def factory(device):
        model = ORIGINAL_FACTORY(device)
        value = dict(model_sha256=state_digest(model.state_dict()))
        path = dest/'initialization.json'
        if path.exists():
            assert read(path) == value
        else:
            atomic_json(path, value)
        if entry['id'] != 'dice':
            assert value == read(root/'runs/dice/initialization.json'), 'Initialization differs'
        return model

    def save(path, payload):
        ORIGINAL_SAVE(path, payload)
        if 'history' in payload and len(payload['history']) == 5:
            value = dict(epoch=5, model_sha256=state_digest(payload['model']))
            if entry['id'] != 'dice':
                assert value == read(root/'runs/dice/warmup.json'), 'First five epochs differ'
            atomic_json(dest/'warmup.json', value)

    def evaluate(model, patients, device, **kwargs):
        result = ORIGINAL_EVALUATE(model, patients, device, **kwargs)
        path = kwargs.get('predictions_path')
        if not path:
            return result
        paired = {name: [] for name in references}
        with np.load(path, allow_pickle=False) as archive:
            for item, row in zip(patients.items, result['cases']):
                assert item['case_name'] == row['case_name']
                truth = item['label'][0].numpy().astype(np.uint8)
                pred = archive[item['case_name']]
                row['anterior_to_posterior'] = int(((truth == 1) & (pred == 2)).sum())
                row['posterior_to_anterior'] = int(((truth == 2) & (pred == 1)).sum())
                assert row['swaps'] == row['anterior_to_posterior'] + row['posterior_to_anterior']
                assert row['missed'] + row['extra'] + row['swaps'] == int((truth != pred).sum())
                for name, volumes in references.items():
                    paired[name].append(paired_counts(truth, volumes[item['case_name']], pred))
        for name in ('anterior_to_posterior', 'posterior_to_anterior'):
            result['summary'][name] = sum(row[name] for row in result['cases'])
        result['paired'] = {name: aggregate(rows) for name, rows in paired.items()}
        return result

    train.objective_components = objectives.components
    train.Targets = objectives.Targets
    train.select_weights = objectives.select_weights
    train.evaluate = evaluate
    train.model_factory = factory
    train.save_checkpoint = save


def checked(root, p, entry):
    record = train.completed_record(root/'runs'/entry['id'], train.identity(p, entry))
    assert record is not None
    return record


def archive_and_retire(root, p, entry):
    dest = root/'runs'/entry['id']
    record = checked(root, p, entry)
    archive = root/'recorded_results'/entry['id']
    archive.mkdir(parents=True, exist_ok=True)
    # These small reference masks remain only while paired comparisons need them.
    if entry['id'] in ('dice', 'winning_constraint'):
        target = root/'references'/f"{entry['id']}.npz"
        target.parent.mkdir(exist_ok=True)
        source = dest/'validation_predictions.npz'
        if source.exists():
            shutil.copyfile(source, target)
        assert digest(target) == (record['artifacts'].get('validation_predictions.npz')
                                  or record['retired_artifacts']['validation_predictions.npz'])
    bulky = ('best.pt', 'validation_predictions.npz')
    for name, expected in record['artifacts'].items():
        if name not in bulky:
            shutil.copyfile(dest/name, archive/name)
            assert digest(archive/name) == expected
    # Record all hashes before unlinking. Resume accepts an interrupted retirement.
    retired = dict(record.get('retired_artifacts', {}))
    for name in bulky:
        if name in record['artifacts']:
            retired[name] = record['artifacts'].pop(name)
    atomic_json(dest/'checkpoint_retired.json', dict(identity=record['identity'], artifacts=retired,
        reason='All selected-checkpoint evaluations and small result artifacts verified before scratch deletion.'))
    record['retired_artifacts'] = retired
    atomic_json(dest/'completion.json', record)
    for name in ('completion.json', 'checkpoint_retired.json'):
        shutil.copyfile(dest/name, archive/name)
        assert digest(dest/name) == digest(archive/name)
    for name, expected in retired.items():
        path = dest/name
        if path.exists():
            assert digest(path) == expected
            path.unlink()
    checked(root, p, entry)


def report(root, p, state, current=None):
    rows = []
    for entry in p['runs']:
        dest = root/'runs'/entry['id']
        if not (dest/'completion.json').exists():
            continue
        record = checked(root, p, entry)
        final = read(dest/'validation_best.json')
        rows.append(dict(id=entry['id'], best_epoch=record['best_epoch'],
            completed_epochs=record['completed_epochs'], training_seconds=record['training_seconds'],
            **record['validation'], paired=final.get('paired', {})))
    ranked = sorted(rows, key=lambda row: (-row['macro_dice'], row['id']))
    atomic_json(root/'results.json', dict(state=state, current=current, completed=len(rows),
        planned=len(p['runs']), job_id=os.environ.get('SLURM_JOB_ID'), runs=rows,
        ranking=[dict(id=r['id'], dice=r['macro_dice']) for r in ranked]))
    if rows:
        columns = ['id', 'macro_dice', 'foreground_dice', 'best_epoch', 'completed_epochs',
                   'missed', 'extra', 'swaps', 'anterior_to_posterior', 'posterior_to_anterior',
                   'surface_distance_mean_mm', 'hd95_mm', 'ap_interface_chamfer_mm']
        path = root/'results.csv'
        with path.with_suffix('.tmp').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows({k:r.get(k) for k in columns} for r in rows)
        path.with_suffix('.tmp').replace(path)


def finish(root, p):
    for entry in p['runs']:
        checked(root, p, entry)
    for name in ('references', 'loss_geometry', 'diagnostic_geometry'):
        path = root/name
        if path.exists():
            assert not path.is_symlink()
            shutil.rmtree(path)
    report(root, p, 'complete')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    stage, root = args.stage, args.root
    for name, sha in read(stage/'freeze.json').items():
        assert digest(stage/name) == sha, name
    for name, sha in read(stage/'source_manifest.json').items():
        assert digest(stage/'source'/name) == sha, name
    p = read(stage/'protocol.json')
    assert p['runs'] == objectives.entries() and len(p['runs']) == 15
    assert p['max_epochs'] == 75 and p['step_size'] == 20 and p['lr_gamma'] == .5
    assert p['early_stopping'] == dict(min_epochs=25, patience=8, min_delta=.0005)
    assert digest(p['pkl']) == p['pkl_sha256']
    assert torch.__version__ == p['runtime_versions']['torch']
    assert monai.__version__ == p['runtime_versions']['monai']
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    torch.set_num_threads(4)
    root.mkdir(parents=True, exist_ok=True)
    with (root/'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        binding = read(stage/'freeze.json')
        if (root/'binding.json').exists():
            assert read(root/'binding.json') == binding
        else:
            atomic_json(root/'binding.json', binding)
        if (root/'results.json').exists() and read(root/'results.json')['state'] == 'complete':
            return
        if (root/'finalization.json').exists():
            finish(root, p)
            return

        def stop(signum, frame):
            global STOP
            STOP = True
        signal.signal(signal.SIGUSR1, stop)
        started = time.monotonic()
        report(root, p, 'preflight')
        if not (root/'gpu_preflight.json').exists():
            from .tests import gpu_preflight
            gpu_preflight(p, root)
        for entry in p['runs']:
            dest = root/'runs'/entry['id']
            if not (dest/'completion.json').exists():
                if STOP or time.monotonic()-started > 23*3600:
                    report(root, p, 'paused_walltime', entry['id'])
                    return
                quota = subprocess.check_output(['beegfs', 'quota', 'list-usage', '--uids',
                    'current', '--pool', 'pool_home'], text=True, stderr=subprocess.DEVNULL)
                match = re.search(r'(\d+(?:\.\d+)?)GiB/(\d+(?:\.\d+)?)GiB', quota)
                if not match or float(match[2])-float(match[1]) < 4:
                    report(root, p, 'paused_storage', entry['id'])
                    return
                report(root, p, 'training', entry['id'])
                try:
                    install(root, entry)
                    train.run(p, root, entry)
                except BaseException as error:
                    atomic_json(root/'failure.json', dict(run=entry['id'], error=repr(error)))
                    report(root, p, 'failed', entry['id'])
                    raise
            record = checked(root, p, entry)
            history = list(csv.DictReader((dest/'metrics.csv').open()))
            best = max(history, key=lambda row: float(row['validation_dice']))
            assert 25 <= len(history) == record['completed_epochs'] <= 75
            assert int(best['epoch']) == record['best_epoch']
            assert abs(float(best['validation_dice'])-record['validation']['macro_dice']) < 1e-7
            for name in ('initialization.json', 'warmup.json'):
                record['artifacts'][name] = digest(dest/name)
            atomic_json(dest/'completion.json', record)
            if entry['id'] == 'winning_constraint':
                expected = p['winner_reference']
                matched = (record['best_epoch'] == expected['best_epoch']
                    and record['completed_epochs'] == expected['completed_epochs']
                    and abs(record['validation']['macro_dice']-expected['macro_dice']) < 1e-7)
                if not matched:
                    atomic_json(root/'failure.json', dict(run=entry['id'], error='Winner replay changed'))
                    report(root, p, 'failed', entry['id'])
                    raise ValueError('Winner replay changed')
                atomic_json(root/'winner_replay.json', dict(status='passed', **expected))
            report(root, p, 'recording', entry['id'])
            archive_and_retire(root, p, entry)
            report(root, p, 'running')
            gc.collect()
            torch.cuda.empty_cache()
        atomic_json(root/'finalization.json', dict(completed=len(p['runs'])))
        finish(root, p)


if __name__ == '__main__':
    main()
