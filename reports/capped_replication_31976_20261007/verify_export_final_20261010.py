"""Audit the completed campaign and export metadata; never train or submit."""
from pathlib import Path
import datetime
import hashlib
import importlib.util
import io
import json
import math
import subprocess
import sys
import tarfile

stage = Path('/mnt/beegfsstudents/home/3160552/hippo/cluster_protocol_20261007/capped_replication_31976')
root = Path('/mnt/beegfsstudents/home/3160552/capped_replication_31976_20261007_results')
ops = stage / 'operations/recovery_20261008'
archive = Path('/tmp/hippo_replication_691005_verified_metadata_20261010.tar.gz')
for path in [stage / 'source', stage / 'source/scripts', stage / 'source/baselines/nnUNet']:
    sys.path.insert(0, str(path))
spec = importlib.util.spec_from_file_location('recovery_worker', ops / 'capped_replication_worker.py')
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)
operations = worker.read(ops / 'operations_manifest.json')
for name, expected in operations['files'].items():
    assert worker.digest(ops / name) == expected
plan = worker.check_frozen(stage, Path('/mnt/beegfsstudents/home/3160552/hippo/datasets/Dataset101_MSD/msd_hippocampus_full.pkl'))
summary = worker.read(root / 'summary.json')
assert summary['status'] == 'complete' and not summary['failures']
assert len(summary['runs']) == summary['completed_new_runs'] == 448
identity = lambda r: tuple(r[k] for k in ('model', 'objective', 'fold', 'seed'))
assert {identity(r) for r in summary['runs']} == {identity(r) for r in plan['runs']}
assert len({identity(r) for r in summary['runs'] + summary['reused_runs']}) == 504
slurm = subprocess.check_output(['sacct', '-j', '691005', '-X', '--format=JobID,State,Elapsed,Start,End,ExitCode', '--parsable2', '--noheader'], text=True)
assert '691005|COMPLETED|' in slurm and '|0:0' in slurm
from scripts.capped_depth_delay import core
from scripts.capped_replication_verify import verify_run
# The worker intentionally discarded .pt files after verifying full checkpoints.
# Audit their retained proofs, and re-run all production checks on retained files.
original_artifacts = dict(core.ARTIFACTS)
core.ARTIFACTS = {k: v for k, v in original_artifacts.items() if not v.endswith('.pt')}
payloads = {}
def include(path, name):
    assert name not in payloads
    payloads[name] = path.read_bytes()

for i, row in enumerate(summary['runs'], 1):
    model, arm, fold, seed = identity(row)
    relative = Path(model) / arm / f'fold{fold}/seed{seed}/run'
    output = root / relative
    stored = worker.verify_stored(output)
    assert identity(stored) == identity(row)
    directory = stage / 'source/experiments/capped_replication_31976_20261007' / model
    audited = verify_run(root, model, arm, directory, fold=fold, seed=seed)
    for key, value in audited.items():
        assert row[key] == value, (identity(row), key)
    manifest = worker.read(output / 'metadata_manifest.json')
    done = worker.read(output / 'completion.json')
    config = worker.read(output / 'config.json')
    for key, name in original_artifacts.items():
        if not name.endswith('.pt'):
            continue
        proof = manifest['discarded_checkpoint_proofs'][name]
        assert proof['file_sha256'] == done[key]
        assert proof['configuration_verified'] and proof['finite_tensors']
        assert len(proof['model_tensor_sha256']) == 64
        expected = done['selected_step'] if name == 'checkpoint_best.pt' else row['updates']
        assert proof['update'] == expected
    for filename, score_key in [('validation_latest.json', 'final_dice'), ('validation_best.json', 'best_dice')]:
        validation = worker.read(output / filename)
        cases = validation['cases']
        assert len(cases) == 52
        for case in cases:
            assert all(math.isfinite(case[k]) and 0 <= case[k] <= 1 for k in ['anterior_hard_dice', 'posterior_hard_dice'])
        mean = sum((c['anterior_hard_dice'] + c['posterior_hard_dice']) / 2 for c in cases) / 52
        assert abs(mean - row[score_key]) < 1e-12
    for p in output.iterdir():
        if p.is_file():
            assert p.suffix in ['.json', '.csv']
            include(p, str(Path('results') / relative / p.name))
    if i % 56 == 0:
        print(json.dumps({'audited_runs': i, 'total': 448}), flush=True)

for name in ['summary.json', 'progress.json', 'results.csv', 'results.md', 'campaign_binding.json']:
    include(root / name, 'results/' + name)
frozen_manifest = worker.read(stage / 'source_manifest.json')
include(stage / 'source_manifest.json', 'frozen/source_manifest.json')
for name in frozen_manifest:
    include(stage / 'source' / name, 'frozen/source/' + name)
for name in ['capped_replication_worker.py', 'run_capped_replication_31976_recovery.sbatch', 'operations_manifest.json', 'staging_receipt.json', 'queue_preflight.json', 'queue_submission_receipt.json']:
    include(ops / name, 'operations/' + name)
for suffix in ['out', 'err']:
    p = stage / f'slurm-hippo-cap-replication-retry-691005.{suffix}'
    if p.exists():
        include(p, 'slurm/' + p.name)
receipt = dict(status='verified', job_id=691005, slurm=slurm.strip(), new_conditions=448,
               retained_pilot_conditions=56, total_conditions=504, failures=0,
               verified_storage_acknowledgments=448, production_metadata_audits=448,
               best_and_final_52_case_validation_means_verified=448,
               strict_optimization_flop_ledgers_verified=448, calibration_excluded_and_reported=True,
               frozen_training_configuration_and_source_verified=True,
               discarded_full_checkpoint_proofs_verified=896,
               checkpoint_retention='verified_metadata_only',
               source_manifest_sha256=worker.digest(stage / 'source_manifest.json'),
               verified_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
payloads['verification.json'] = (json.dumps(receipt, indent=2) + '\n').encode()
export = {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()}
payloads['export_manifest.json'] = (json.dumps(export, indent=2) + '\n').encode()
with tarfile.open(archive, 'w:gz') as tar:
    for name, data in payloads.items():
        info = tarfile.TarInfo(name)
        info.size = len(data)
        info.mode = 0o600
        tar.addfile(info, io.BytesIO(data))
print(json.dumps({'status': 'verified_export_ready', 'archive': str(archive),
                  'sha256': worker.digest(archive), 'bytes': archive.stat().st_size,
                  'export_manifest_sha256': hashlib.sha256(payloads['export_manifest.json']).hexdigest(),
                  'verification': receipt}), flush=True)
