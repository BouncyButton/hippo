"""Verified local-to-Drive transport and completed-run retirement."""
from pathlib import Path
import csv
import json
import math
import time
from scripts.colab_medsam3.runtime import read, write, digest, copy_verified
from scripts.colab_adni.runtime import restore_recovery as restore_snapshot


def publish_metadata(output, durable):
    for path in sorted(output.iterdir()):
        if path.is_file() and path.suffix in ('.json', '.csv', '.log', '.jsonl', '.png'):
            if path.name not in ('snapshot_manifest.json', 'supervisor_ready.json'):
                copy_verified(path, durable/path.name)


def verify_archive(durable):
    manifest = read(durable/'archive_manifest.json')
    for name, expected in manifest.items():
        if Path(name).name != name or digest(durable/name) != expected:
            raise ValueError('Archive mismatch: '+name)
        if name.endswith('.json'):
            read(durable/name)
    return read(durable/'result.json')


def archive(output, durable, kind, model, fold, seed, protocol):
    done = read(output/'completion.json')
    if done['status'] != 'complete':
        raise ValueError('Cannot accept smoke or partial runs')
    if kind == 'baselines':
        p = read(protocol/'protocol.json')
        schedule = p['models'][p['trainer_alias']]['objectives']['dice']
        if done['updates'] != schedule['max_steps'] or done['discarded_attempts']:
            raise ValueError('Final model has fewer successful updates than prescribed; audit, do not classify seed')
        for key, filename in (('budget_ledger_sha256', 'budget_ledger.json'), ('metrics_sha256', 'metrics.csv')):
            if digest(output/filename) != done[key]:
                raise ValueError('Completion binding mismatch')
        evaluation = read(output/'validation_latest.json')
        with (output/'metrics.csv').open() as stream:
            history = list(csv.DictReader(stream))
        if not history or int(history[-1]['charged_attempts']) != done['training_attempts_charged']:
            raise ValueError('Validation is not from the final charged update')
        if not math.isclose(float(history[-1]['macro_hard_dice']), evaluation['summary']['macro_hard_dice'], abs_tol=1e-8):
            raise ValueError('Final validation disagrees with bound metric history')
        expected = set(read(protocol/'splits_10patients.json')[fold]['val'])
        if {r['case_name'] for r in evaluation['cases']} != expected or len(evaluation['cases']) != 10:
            raise ValueError('Incomplete validation')
        if len(read(output/'training_final.json')['cases']) != 10:
            raise ValueError('Incomplete training evaluation')
        score = evaluation['summary']['macro_hard_dice']
        budget = done['consumed_training_flops']
    else:
        p = read(protocol)
        evaluation = read(output/'evaluation.json')
        if done['updates'] != 200 or len(read(output/'training_history.json')) != 200:
            raise ValueError('Expected exactly 200 completed updates')
        if digest(output/'evaluation.json') != done['evaluation_sha256']:
            raise ValueError('Evaluation binding mismatch')
        expected = {(split, case) for split, names in [('training_support', p['train_cases']),
                    ('validation', p['evaluation_cases'])] for case in names}
        if {(r['split'], r['case_name']) for r in evaluation['records']} != expected or len(evaluation['records']) != 20:
            raise ValueError('Incomplete MedSAM3 evaluation')
        score, budget = done['validation_dice'], None
    if not 0 <= score <= 1:
        raise ValueError('Invalid Dice')
    result = dict(model=model, fold=fold, seed=seed, validation_dice=score,
        meets_50_percent=score >= .5, updates=done['updates'], optimization_flops=budget,
        checkpoint='final_budget' if kind == 'baselines' else 'final_update',
        metric='Mean per-hippocampus foreground Dice on the localized 1x2x1 mm crop grid',
        ap_swaps_applicable=False, evaluation_scope='exploratory_validation_selection')
    write(output/'result.json', result)
    publish_metadata(output, durable)
    manifest = {p.name: digest(p) for p in sorted(durable.iterdir()) if p.is_file()
        and p.suffix in ('.json', '.csv', '.log', '.jsonl', '.png')
        and p.name not in ('archive_manifest.json', 'cleanup.json', 'invocation.json')}
    write(durable/'archive_manifest.json', manifest)
    verify_archive(durable)
    deleted = []
    for folder in (output, durable):
        for path in folder.glob('checkpoint_*.pt'):
            key = 'checkpoint_sha256' if kind != 'baselines' else (
                'checkpoint_latest_sha256' if path.name == 'checkpoint_latest.pt' else 'checkpoint_best_sha256')
            sha = digest(path)
            if sha != done[key]:
                raise ValueError('Checkpoint changed; retained')
            deleted.append(dict(path=str(path), bytes=path.stat().st_size, sha256=sha))
            path.unlink()
    marker = durable/'recovery/latest.json'
    if marker.exists():
        pointer = read(marker)
        name = pointer['file']
        if Path(name).name != name:
            raise ValueError('Unsafe recovery filename')
        path = marker.parent/name
        if digest(path) != pointer['sha256']:
            raise ValueError('Recovery changed; retained')
        deleted.append(dict(path=str(path), bytes=path.stat().st_size, sha256=pointer['sha256']))
        path.unlink()
        marker.unlink()
    write(durable/'cleanup.json', dict(status='complete', verified_at=time.time(), deleted=deleted))
    return result
