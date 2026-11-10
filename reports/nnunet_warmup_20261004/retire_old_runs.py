"""Preserve records, then retire only the explicitly authorized campaign's nnU-Net runs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile

CAMPAIGN = Path('/mnt/beegfsstudents/home/3160552/compute_matched_baselines_681342')
TARGET = CAMPAIGN / 'nnunet'
BACKUP = Path('/mnt/beegfsstudents/home/3160552/hippo/cluster_protocol_20261004/nnunet_warmup/retired_681342')
ARTIFACTS = {'config_sha256': 'config.json', 'checkpoint_best_sha256': 'checkpoint_best.pt',
             'checkpoint_latest_sha256': 'checkpoint_latest.pt', 'metrics_sha256': 'metrics.csv',
             'budget_ledger_sha256': 'budget_ledger.json', 'validation_best_sha256': 'validation_best.json'}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def validate_target():
    assert TARGET.parent == CAMPAIGN and TARGET.name == 'nnunet'
    assert TARGET.is_dir() and not TARGET.is_symlink()
    assert TARGET.stat().st_uid == os.getuid()
    assert not any(p.is_symlink() for p in TARGET.rglob('*'))


def verify_archive(manifest):
    archive = BACKUP / 'old_nnunet_records.tar.gz'
    assert digest(archive) == manifest['archive_sha256']
    expected = manifest['archived_records']
    with tarfile.open(archive) as tar:
        actual = {m.name for m in tar.getmembers() if m.isfile()}
        assert actual == set(expected)
        for name, record in expected.items():
            with tar.extractfile(name) as stream:
                assert hashlib.sha256(stream.read()).hexdigest() == record['sha256']


def archive_records():
    validate_target()
    BACKUP.mkdir(parents=True, exist_ok=True)
    manifest_path = BACKUP / 'retirement_manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        verify_archive(manifest)
        print(json.dumps({'archive': manifest['archive_path'], 'status': 'already_verified'}))
        return
    completions = []
    for fold in range(3):
        for seed in range(3):
            run = TARGET / f'fold{fold}' / f'seed{seed}' / 'run'
            done = json.loads((run / 'completion.json').read_text())
            assert done['status'] == 'complete'
            assert (done['model'], done['fold'], done['seed']) == ('nnunet', fold, seed)
            assert done['training_attempts_charged'] == 327
            assert done['consumed_training_flops'] <= 116259000000000
            for key, filename in ARTIFACTS.items():
                assert digest(run / filename) == done[key]
            completions.append({'fold': fold, 'seed': seed, **done})
    archived, inventory = {}, {}
    files = []
    for path in sorted(TARGET.rglob('*')):
        if not path.is_file():
            continue
        relative = str(path.relative_to(CAMPAIGN))
        stat = path.stat()
        inventory[relative] = {'size_bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
        if path.name not in ('checkpoint_best.pt', 'checkpoint_latest.pt'):
            files.append((path, relative))
    for name in ('summary.json', 'summary.csv', 'summary.md', 'source_manifest.json'):
        path = CAMPAIGN / name
        if path.exists():
            files.append((path, 'campaign_summary/' + name))
    files.append((CAMPAIGN / 'source/experiments/compute_matched_baselines_20261003/protocol.json',
                  'campaign_summary/original_protocol.json'))
    archive = BACKUP / 'old_nnunet_records.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        for path, relative in files:
            archived[relative] = {'size_bytes': path.stat().st_size, 'sha256': digest(path)}
            tar.add(path, arcname=relative)
    manifest = {'original_campaign': 681342, 'target_directory': str(TARGET),
                'archive_path': str(archive), 'archive_sha256': digest(archive),
                'archived_records': archived, 'original_file_inventory': inventory,
                'original_completions': completions, 'status': 'records_preserved',
                'preserved_at_utc': datetime.now(timezone.utc).isoformat(),
                'authorization': 'User explicitly requested deleting old nnU-Net runs and all nine warmup reruns.',
                'note': 'Original metrics and spent compute retained. Old checkpoints will be deleted; original U-Net/SwinUNETR and frozen source are outside the deletion target.'}
    verify_archive(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    seconds = sum(c['seconds_this_invocation'] for c in completions)
    print(json.dumps({'archive': str(archive), 'archive_bytes': archive.stat().st_size,
                      'archived_files': len(archived), 'checkpoint_files_to_delete': 18,
                      'original_nine_run_seconds': seconds, 'manifest': str(manifest_path)}))


def delete_runs():
    manifest_path = BACKUP / 'retirement_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    assert manifest['target_directory'] == str(TARGET) and manifest['original_campaign'] == 681342
    verify_archive(manifest)
    if manifest['status'] == 'deleted':
        assert not TARGET.exists()
        print(json.dumps({'status': 'already_deleted', 'target': str(TARGET)}))
        return
    validate_target()
    actual = {}
    for path in TARGET.rglob('*'):
        if path.is_file():
            stat = path.stat()
            actual[str(path.relative_to(CAMPAIGN))] = {'size_bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
    assert actual == manifest['original_file_inventory'], 'Old runs changed since preservation'
    for relative, record in manifest['archived_records'].items():
        if relative.startswith('nnunet/'):
            assert digest(CAMPAIGN / relative) == record['sha256']
    assert (CAMPAIGN / 'unet').is_dir() and (CAMPAIGN / 'swinunetr').is_dir()
    shutil.rmtree(TARGET)
    assert not TARGET.exists()
    manifest['status'] = 'deleted'
    manifest['deleted_at_utc'] = datetime.now(timezone.utc).isoformat()
    manifest['deleted_bytes'] = sum(r['size_bytes'] for r in actual.values())
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    (CAMPAIGN / 'NNUNET_RETIRED.json').write_text(json.dumps({
        'original_campaign': 681342, 'original_summary_is_historical': True,
        'retired_model': 'nnunet', 'retirement_manifest': str(manifest_path),
        'revision_protocol': 'nnunet_warmup_20261004',
        'deleted_at_utc': manifest['deleted_at_utc']}, indent=2) + '\n')
    print(json.dumps({'status': 'deleted', 'target': str(TARGET),
                      'bytes_removed': manifest['deleted_bytes'], 'preserved_records': len(manifest['archived_records'])}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('archive', 'delete'))
    phase = parser.parse_args().phase
    archive_records() if phase == 'archive' else delete_runs()
