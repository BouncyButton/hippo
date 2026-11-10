"""One allocation, sequential conditions, verified lossless model export."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

from campaign90 import BUDGET, MODELS, digest, write_json
from scripts.capped_depth_delay.constants import CAMPAIGN
from scripts.capped_depth_delay.preflight import check_frozen
from scripts.capped_depth_delay.verify import verify_run


def tensor_digest(state):
    hasher = hashlib.sha256()
    for name, value in state.items():
        hasher.update(name.encode())
        hasher.update(value.detach().cpu().contiguous().numpy().tobytes())
    return hasher.hexdigest()


def verify_calibration(output, directory, arm):
    reference = json.loads((directory / 'calibration_references.json').read_text())[arm]
    actual = json.loads((output / 'calibration.json').read_text())
    if actual['anchor_model_sha256'] != reference['anchor_model_sha256']:
        raise ValueError('Native calibration-prefix model differs from the original')
    if not reference.get('anchor_only'):
        for name, weight in reference['weights'].items():
            if actual['weights'][name] != weight or actual['statistics'][name] != reference['statistics'][name]:
                raise ValueError('Frozen existing coefficient or statistics changed')
        for row, old in zip(actual['cases'], reference['cases']):
            if any(row[key] != value for key, value in old.items()):
                raise ValueError('Existing calibration observations changed')
    return True


def export_condition(output, export, row):
    import torch
    export.mkdir(parents=True, exist_ok=True)
    for path in output.iterdir():
        if path.is_file() and path.suffix in ('.json', '.csv'):
            shutil.copyfile(path, export / path.name)
    model_hashes = {}
    for source_name, filename in [('checkpoint_best.pt', 'checkpoint_best_model.pt'),
                                  ('checkpoint_latest.pt', 'checkpoint_final_model.pt')]:
        payload = torch.load(output / source_name, map_location='cpu', weights_only=False)
        expected = tensor_digest(payload['model'])
        # Clone individual tensors to avoid saving unrelated backing storages.
        state = {name: value.detach().cpu().clone().contiguous() for name, value in payload['model'].items()}
        if tensor_digest(state) != expected:
            raise ValueError('Export changed model tensor values')
        lean = dict(model=state, configuration=payload['configuration'],
                    step=payload['step'], score=payload.get('score'),
                    model_tensor_sha256=expected, source_checkpoint_sha256=digest(output / source_name),
                    source_completion_sha256=digest(output / 'completion.json'),
                    export_policy='Lossless model-only checkpoint. Resume/optimizer state is transient; exact original model tensors retained.')
        torch.save(lean, export / filename)
        reloaded = torch.load(export / filename, map_location='cpu', weights_only=False)
        if tensor_digest(reloaded['model']) != expected:
            raise ValueError('Saved model export verification failed')
        model_hashes[filename] = expected
        del payload, state, lean, reloaded
    write_json(export / 'condition_summary.json', row)
    files = {p.name: digest(p) for p in export.iterdir() if p.is_file()}
    write_json(export / 'export_manifest.json', dict(files=files, model_tensor_sha256=model_hashes,
        original_artifacts_verified_before_export=True,
        original_completion_sha256=row['completion_sha256']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--scratch', type=Path, required=True)
    parser.add_argument('--pkl', type=Path, required=True)
    args = parser.parse_args()
    root, scratch = args.root, args.scratch
    plan, protocols = check_frozen(root, args.pkl)
    preflight = json.loads((root / 'preflight.json').read_text())
    if (preflight['status'] != 'passed'
            or preflight['plan_sha256'] != digest(root / 'source/experiments' / CAMPAIGN / 'plan.json')
            or preflight['source_manifest_sha256'] != digest(root / 'source_manifest.json')):
        raise ValueError('Bound preflight missing')
    scratch.mkdir(parents=True, exist_ok=True, mode=0o700)
    if scratch.stat().st_uid != os.getuid() or not str(scratch).startswith('/tmp/hippo-capped-depth-delay-'):
        raise ValueError('Unexpected scratch owner/path')
    work = scratch / 'work'
    work.mkdir(exist_ok=True)
    if not (work / 'source').exists():
        (work / 'source').symlink_to(root / 'source', target_is_directory=True)
    (root / 'outbox').mkdir(exist_ok=True)
    (root / 'acks').mkdir(exist_ok=True)
    baseline_campaign = root / 'source/reports/four_models_six_optimization_31976_20261006/summary.json'
    baseline_rows = json.loads(baseline_campaign.read_text())['runs']
    baselines = {r['model']: r for r in baseline_rows if r['objective'] == 'dice'}
    summary_path = root / 'summary.json'
    rows = json.loads(summary_path.read_text())['runs'] if summary_path.exists() else []
    completed = {(r['model'], r['objective']) for r in rows}
    try:
        for index, entry in enumerate(plan['runs'], 1):
            identity, arm = entry['model'], entry['objective']
            if (identity, arm) in completed:
                continue
            write_json(root / 'progress.json', dict(status='training', completed=len(rows), failed=0,
                total=44, current_index=index, model=identity, objective=arm))
            directory = root / 'source/experiments' / CAMPAIGN / identity
            output = work / identity / arm / 'fold0/seed0/run'
            subprocess.run([sys.executable, '-m', 'scripts.capped_depth_delay.train',
                '--model', MODELS[identity]['alias'], '--objective', arm, '--fold', '0', '--seed', '0',
                '--protocol-dir', str(directory), '--output', str(output), '--pkl', str(args.pkl),
                '--threads', '4', '--resume'], check=True)
            row = verify_run(work, identity, arm, directory)
            row.update(calibration_anchor_and_existing_coefficients_exact=verify_calibration(output, directory, arm),
                       gain_vs_model_baseline_pp=100 * (row['final_dice'] - baselines[identity]['final_dice']),
                       sdf_first_update=entry['sdf_first_update'], calibration_update=entry['calibration_update'],
                       ramp_updates=entry['ramp_updates'])
            lean = scratch / 'exports' / f'{index:02d}_{identity}_{arm}'
            export_condition(output, lean, row)
            run_id = lean.name
            archive = root / 'outbox' / (run_id + '.tar')
            partial = archive.with_suffix('.part')
            if shutil.disk_usage(root).free < 2 * sum(p.stat().st_size for p in lean.iterdir()):
                raise RuntimeError('Insufficient export staging disk')
            with tarfile.open(partial, 'w') as tar:
                for path in sorted(lean.iterdir()):
                    tar.add(path, arcname=path.name, recursive=False)
            partial.replace(archive)
            ready = dict(run_id=run_id, index=index, model=identity, objective=arm,
                         archive_name=archive.name, archive_sha256=digest(archive),
                         archive_bytes=archive.stat().st_size,
                         export_manifest_sha256=digest(lean / 'export_manifest.json'))
            write_json(root / 'ready.json', ready)
            write_json(root / 'progress.json', dict(status='awaiting_verified_transfer',
                completed=len(rows), failed=0, total=44, current_index=index,
                model=identity, objective=arm, archive_bytes=ready['archive_bytes']))
            deadline = time.monotonic() + 3600
            ack = root / 'acks' / (run_id + '.json')
            while not ack.exists():
                if time.monotonic() > deadline:
                    raise RuntimeError('Verified export transfer not acknowledged within one hour')
                time.sleep(10)
            receipt = json.loads(ack.read_text())
            if (receipt['archive_sha256'] != ready['archive_sha256']
                    or receipt['run_id'] != run_id or not receipt['all_files_sha256_verified']):
                raise ValueError('Export acknowledgment differs')
            destination = root / identity / arm / 'fold0/seed0/run'
            destination.mkdir(parents=True, exist_ok=True)
            for path in lean.iterdir():
                if path.suffix in ('.json', '.csv'):
                    shutil.copyfile(path, destination / path.name)
            write_json(destination / 'local_model_storage_receipt.json', receipt)
            archive.unlink()
            (root / 'ready.json').unlink()
            shutil.rmtree(lean)
            shutil.rmtree(output.parent.parent.parent)
            rows.append(row)
            completed.add((identity, arm))
            write_json(summary_path, dict(status='running', runs=rows, failures=[], total=44,
                budget_flops=BUDGET, budget_scope='optimization_only', fold=0, seed=0))
            write_json(root / 'progress.json', dict(status='running', completed=len(rows), failed=0, total=44))
            print(json.dumps({'condition_complete': row}), flush=True)
        check_frozen(root, args.pkl)
        write_json(summary_path, dict(status='complete', runs=rows, failures=[], total=44,
            budget_flops=BUDGET, budget_scope='optimization_only', fold=0, seed=0))
        write_json(root / 'progress.json', dict(status='complete', completed=44, failed=0, total=44))
    except Exception as error:
        write_json(root / 'progress.json', dict(status='failed', completed=len(rows), failed=1, total=44,
            error=repr(error), current_index=index, model=identity, objective=arm))
        write_json(summary_path, dict(status='failed', runs=rows, failures=[repr(error)], total=44,
            budget_flops=BUDGET, budget_scope='optimization_only'))
        raise


if __name__ == '__main__':
    main()
