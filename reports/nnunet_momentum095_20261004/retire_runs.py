"""Preserve completed job 682319 records, then retire only its three nnU-Net runs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

JOB = 682319
ROOT = Path('/mnt/beegfsstudents/home/3160552/nnunet_momentum095_682319')
TARGET = ROOT / 'nnunet'
STAGE = Path('/mnt/beegfsstudents/home/3160552/hippo/cluster_protocol_20261004/nnunet_momentum095')
BACKUP = STAGE / 'retired_682319'
MANIFEST = BACKUP / 'retirement_manifest.json'
ARCHIVE = BACKUP / 'run_records.tar.gz'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def stopped():
    state = subprocess.check_output(['sacct', '-X', '-j', str(JOB), '-n', '-P',
                                     '--format=JobID,State,ExitCode,Elapsed,End'], text=True).strip()
    parts = state.split('|')
    assert parts[0] == str(JOB) and parts[1] == 'COMPLETED' and parts[2] == '0:0', state
    queue = subprocess.check_output(['squeue', '-h', '-u', '3160552', '-o', '%i'], text=True)
    assert str(JOB) not in queue.split(), 'Job is still active'
    return state


def target_inventory():
    assert TARGET.parent == ROOT and TARGET.name == 'nnunet'
    assert TARGET.is_dir() and not TARGET.is_symlink() and TARGET.stat().st_uid == os.getuid()
    assert not any(p.is_symlink() for p in TARGET.rglob('*'))
    runs = sorted(TARGET.glob('fold*/seed*/run'))
    assert {str(p.relative_to(TARGET)) for p in runs} == {f'fold0/seed{s}/run' for s in (0, 1, 2)}
    return {str(f.relative_to(ROOT)): {'size': f.stat().st_size, 'mtime_ns': f.stat().st_mtime_ns}
            for f in TARGET.rglob('*') if f.is_file()}


def verify_archive(manifest):
    assert digest(ARCHIVE) == manifest['archive_sha256']
    with tarfile.open(ARCHIVE) as tar:
        assert {m.name for m in tar.getmembers() if m.isfile()} == set(manifest['records'])
        for name, record in manifest['records'].items():
            assert hashlib.sha256(tar.extractfile(name).read()).hexdigest() == record['sha256'], name


def archive():
    state = stopped()
    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text()); verify_archive(manifest)
        print(json.dumps({'status': manifest['status'], 'archive': str(ARCHIVE)})); return
    inventory = target_inventory()
    completions = []
    for seed in (0, 1, 2):
        run = TARGET / f'fold0/seed{seed}/run'
        done = json.loads((run / 'completion.json').read_text())
        config = json.loads((run / 'config.json').read_text())
        assert done['status'] == 'complete' and done['fold'] == 0 and done['seed'] == seed
        assert done['updates'] == done['training_attempts_charged'] == 327 and done['discarded_attempts'] == 0
        assert done['consumed_training_flops'] <= done['budget_flops'] == 116259000000000
        assert config['nnunet_sgd_momentum'] == .95 and config['nnunet_warmup_fraction'] == 0.0
        for key, name in {'config_sha256': 'config.json', 'checkpoint_best_sha256': 'checkpoint_best.pt',
                          'checkpoint_latest_sha256': 'checkpoint_latest.pt', 'metrics_sha256': 'metrics.csv',
                          'budget_ledger_sha256': 'budget_ledger.json', 'validation_best_sha256': 'validation_best.json'}.items():
            assert digest(run / name) == done[key], name
        completions.append(done)
    summary = json.loads((ROOT / 'summary.json').read_text())
    assert {(r['fold'], r['seed']) for r in summary['runs']} == {(0, s) for s in (0, 1, 2)}
    files = [(ROOT / name, name) for name in inventory if not name.endswith('.pt')]
    for name in ('summary.json', 'summary.csv', 'summary.md', 'source_manifest.json'):
        files.append((ROOT / name, 'campaign/' + name))
    for name in ('submission.json', 'slurm-hippo-nnunet-m095-682319.out', 'slurm-hippo-nnunet-m095-682319.err'):
        files.append((STAGE / name, 'scheduler/' + name))
    for f in (ROOT / 'source/experiments/nnunet_momentum095_20261004').iterdir():
        if f.is_file(): files.append((f, 'protocol/' + f.name))
    for name in ('train_compute_matched_baselines.py', 'summarize_compute_matched_baselines.py', 'run_nnunet_momentum095.sbatch'):
        files.append((ROOT / 'source/scripts' / name, 'source/' + name))
    BACKUP.mkdir(exist_ok=True)
    records = {}
    with tarfile.open(ARCHIVE, 'w:gz') as tar:
        for f, name in files:
            assert f.is_file() and not f.is_symlink(), str(f)
            records[name] = {'size': f.stat().st_size, 'sha256': digest(f)}
            tar.add(f, arcname=name)
    manifest = {'job_id': JOB, 'slurm_state': state, 'target': str(TARGET), 'status': 'preserved',
                'archive': str(ARCHIVE), 'archive_sha256': digest(ARCHIVE), 'records': records,
                'file_inventory': inventory, 'completions': completions,
                'summary': summary, 'preserved_at_utc': datetime.now(timezone.utc).isoformat(),
                'authorization': 'User requested deletion of these completed three-seed runs ahead of a new budget.'}
    verify_archive(manifest)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'status': 'preserved', 'archive': str(ARCHIVE), 'records': len(records),
                      'completed_runs': len(completions), 'target_bytes': sum(v['size'] for v in inventory.values())}))


def delete():
    stopped()
    manifest = json.loads(MANIFEST.read_text())
    assert manifest['job_id'] == JOB and manifest['target'] == str(TARGET)
    verify_archive(manifest)
    if manifest['status'] == 'deleted':
        assert not TARGET.exists(); print('Already deleted'); return
    assert target_inventory() == manifest['file_inventory'], 'Runs changed after preservation'
    for relative, record in manifest['records'].items():
        if relative.startswith('nnunet/'):
            assert digest(ROOT / relative) == record['sha256'], relative
    shutil.rmtree(TARGET)
    assert not TARGET.exists()
    manifest.update(status='deleted', deleted_at_utc=datetime.now(timezone.utc).isoformat(),
                    deleted_bytes=sum(v['size'] for v in manifest['file_inventory'].values()))
    MANIFEST.write_text(json.dumps(manifest, indent=2) + '\n')
    (ROOT / 'NNUNET_RETIRED.json').write_text(json.dumps({
        'job_id': JOB, 'state': 'COMPLETED', 'do_not_resume': True,
        'run_directory_deleted': str(TARGET), 'preserved_records': str(BACKUP),
        'retirement_manifest': str(MANIFEST)}, indent=2) + '\n')
    print(json.dumps({'status': 'deleted', 'target': str(TARGET), 'deleted_bytes': manifest['deleted_bytes']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('archive', 'delete'))
    args = parser.parse_args()
    archive() if args.phase == 'archive' else delete()
