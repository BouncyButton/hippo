"""Preserve the analysed, cancelled job 682232; delete only its nnU-Net run directory."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

ROOT = Path('/mnt/beegfsstudents/home/3160552/nnunet_warmup_682232')
TARGET = ROOT / 'nnunet'
STAGE = Path('/mnt/beegfsstudents/home/3160552/hippo/cluster_protocol_20261004/nnunet_warmup')
BACKUP = STAGE / 'retired_682232'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def stopped():
    state = subprocess.check_output(['sacct', '-X', '-j', '682232', '-n', '-P',
                                     '--format=State%40,ExitCode,Elapsed,End'], text=True).strip()
    assert state.startswith('CANCELLED'), state
    queue = subprocess.check_output(['squeue', '-h', '-u', '3160552', '-o', '%i'], text=True)
    assert '682232' not in queue.split(), 'Job still active'
    return state


def validate_target():
    assert TARGET.parent == ROOT and TARGET.name == 'nnunet'
    assert TARGET.is_dir() and not TARGET.is_symlink() and TARGET.stat().st_uid == os.getuid()
    assert not any(p.is_symlink() for p in TARGET.rglob('*'))


def verify_archive(manifest):
    path = BACKUP / 'stopped_run_records.tar.gz'
    assert digest(path) == manifest['archive_sha256']
    with tarfile.open(path) as tar:
        expected = manifest['records']
        assert {m.name for m in tar.getmembers() if m.isfile()} == set(expected)
        for name, value in expected.items():
            with tar.extractfile(name) as stream:
                assert hashlib.sha256(stream.read()).hexdigest() == value['sha256']


def archive():
    state = stopped()
    validate_target()
    audit = json.loads((ROOT / 'stopped_checkpoint_audit.json').read_text())
    assert len(audit['checkpoints']) >= 8, 'Analyse all four saved best/latest pairs before retirement'
    assert any(c['fold'] == 0 and c['seed'] == 1 and c['checkpoint'] == 'latest' for c in audit['checkpoints'])
    BACKUP.mkdir(parents=True, exist_ok=True)
    path = BACKUP / 'deletion_manifest.json'
    if path.exists():
        manifest = json.loads(path.read_text()); verify_archive(manifest)
        print(json.dumps({'status': 'already_preserved', 'archive': manifest['archive']})); return
    files, inventory, completions = [], {}, []
    for run in sorted(TARGET.glob('fold*/seed*/run')):
        done_path = run / 'completion.json'
        if done_path.exists():
            done = json.loads(done_path.read_text())
            assert done['status'] == 'complete' and done['training_attempts_charged'] == 327
            assert done['consumed_training_flops'] <= 116259000000000
            for key, name in {'config_sha256': 'config.json', 'checkpoint_best_sha256': 'checkpoint_best.pt',
                              'checkpoint_latest_sha256': 'checkpoint_latest.pt', 'metrics_sha256': 'metrics.csv',
                              'budget_ledger_sha256': 'budget_ledger.json', 'validation_best_sha256': 'validation_best.json'}.items():
                assert digest(run / name) == done[key]
            completions.append(done)
    for f in sorted(TARGET.rglob('*')):
        if not f.is_file(): continue
        relative = str(f.relative_to(ROOT)); stat = f.stat()
        inventory[relative] = {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
        if f.name not in ('checkpoint_best.pt', 'checkpoint_latest.pt'):
            files.append((f, relative))
    for name in ('stopped_checkpoint_audit.json', 'stopped_failure_records.json', 'source_manifest.json', 'progress_audit_latest.json'):
        f = ROOT / name
        if f.exists(): files.append((f, 'analysis/' + name))
    for name in ('slurm-hippo-nnunet-warmup-682232.out', 'slurm-hippo-nnunet-warmup-682232.err', 'submission.json'):
        f = STAGE / name
        if f.exists(): files.append((f, 'scheduler/' + name))
    for f in (ROOT / 'source/experiments/nnunet_warmup_20261004').iterdir():
        if f.is_file(): files.append((f, 'protocol/' + f.name))
    files.append((ROOT / 'source/scripts/train_compute_matched_baselines.py', 'protocol/frozen_trainer.py'))
    records = {}; tar_path = BACKUP / 'stopped_run_records.tar.gz'
    with tarfile.open(tar_path, 'w:gz') as tar:
        for f, name in files:
            records[name] = {'size': f.stat().st_size, 'sha256': digest(f)}
            tar.add(f, arcname=name)
    manifest = {'job_id': 682232, 'cancelled_state': state, 'target': str(TARGET), 'status': 'preserved',
                'archive': str(tar_path), 'archive_sha256': digest(tar_path), 'records': records,
                'file_inventory': inventory, 'completions': completions, 'analysed_checkpoint_entries': len(audit['checkpoints']),
                'preserved_at_utc': datetime.now(timezone.utc).isoformat(),
                'authorization': 'User requested immediate cancellation, failure analysis, then deletion of these reruns.'}
    verify_archive(manifest)
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'archive': str(tar_path), 'records': len(records), 'completed_runs': len(completions),
                      'archive_bytes': tar_path.stat().st_size, 'checkpoint_entries_analysed': len(audit['checkpoints'])}))


def delete():
    stopped()
    path = BACKUP / 'deletion_manifest.json'; manifest = json.loads(path.read_text())
    assert manifest['job_id'] == 682232 and manifest['target'] == str(TARGET)
    verify_archive(manifest)
    if manifest['status'] == 'deleted':
        assert not TARGET.exists(); print('Already deleted'); return
    validate_target()
    actual = {str(f.relative_to(ROOT)): {'size': f.stat().st_size, 'mtime_ns': f.stat().st_mtime_ns}
              for f in TARGET.rglob('*') if f.is_file()}
    assert actual == manifest['file_inventory'], 'Runs changed after preservation'
    for relative, record in manifest['records'].items():
        if relative.startswith('nnunet/'): assert digest(ROOT / relative) == record['sha256']
    shutil.rmtree(TARGET)
    assert not TARGET.exists()
    manifest.update(status='deleted', deleted_at_utc=datetime.now(timezone.utc).isoformat(),
                    deleted_bytes=sum(v['size'] for v in actual.values()))
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    (ROOT / 'NNUNET_RETIRED.json').write_text(json.dumps({
        'job_id': 682232, 'state': 'CANCELLED', 'do_not_resume': True,
        'run_directory_deleted': str(TARGET), 'preserved_failure_analysis': str(BACKUP),
        'deletion_manifest': str(path)}, indent=2) + '\n')
    print(json.dumps({'status': 'deleted', 'target': str(TARGET), 'deleted_bytes': manifest['deleted_bytes']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('archive', 'delete'))
    phase = parser.parse_args().phase
    archive() if phase == 'archive' else delete()
