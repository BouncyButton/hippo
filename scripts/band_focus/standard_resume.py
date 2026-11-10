"""Restrict the frozen 694515 campaign to its unchanged standard75 arms.

The original protocol/source hashes remain authoritative for checkpoint identity.
This amendment removes exactly three arms; it does not change loss or training code.
"""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT_NAME = 'band_focus_fold0_seed0_5pct_20261011_02'
PROTOCOL_SHA = '03c8a009c9c5484a7dc76cdd31ea7b59a08f5f03129788b4289e08ca9d2a142e'
EXCLUDED = ('long_dice', 'long_bands', 'long_focus25')
SCOPE = 'standard75_scope.json'
ADDED = ('source/scripts/band_focus/standard_resume.py',
         'source/scripts/band_focus/run_standard.sh')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def standard_protocol(original):
    p = copy.deepcopy(original)
    removed = [e for e in p['runs'] if e['id'] in EXCLUDED]
    assert {e['id'] for e in removed} == set(EXCLUDED)
    assert all(e['family'] == 'training_schedule' for e in removed)
    p['runs'] = [e for e in p['runs'] if e['id'] not in EXCLUDED]
    assert len(p['runs']) == len({e['id'] for e in p['runs']}) == 190
    assert all(not e.get('training') and e['family'] != 'training_schedule' for e in p['runs'])
    assert all(e['fraction'] == '5pct' and e['seed'] == 0 for e in p['runs'])
    assert p['fold'] == p['seed'] == 0 and not p['auxiliary_branch']
    assert p['max_epochs'] == 75 and p['step_size'] == 20 and p['lr_gamma'] == .5
    assert p['early_stopping'] == dict(min_epochs=25, patience=8, min_delta=.0005)
    p['main_runs'] = len(p['runs'])
    p['loss_definitions']['schedule'] = 'All 190 active arms use max75, StepLR20 gamma0.5, min25/patience8/delta0.0005.'
    p['interpretation'] = ('User scope amendment, 2026-10-11: only the 190 unchanged standard75 arms remain active. '
        'The three extended150/StepLR40 runs were deleted at the user\'s request. '
        'Ordinary SwinUNETR, 5%, fold0 seed0; all retained configurations, calibration and checkpoint identities are unchanged. '
        'Reused validation is development data; confirm selected losses on new seeds/folds before generalization claims.')
    return p


def verify_payload(root):
    assert os.getuid() == 1395 and root.name == ROOT_NAME and not root.is_symlink()
    assert sha(root/'protocol.json') == PROTOCOL_SHA
    for name, expected in json.loads((root/'payload_manifest.json').read_text()).items():
        if sha(root/name) != expected:
            raise ValueError('Frozen payload changed: ' + name)
    return json.loads((root/'protocol.json').read_text())


def validate_gates(root, p):
    gpu = json.loads((root/'cuda_preflight.json').read_text())
    calibration = json.loads((root/'calibration_preflight.json').read_text())
    assert gpu['passed'] and gpu['parameters_updated'] and gpu['ordinary_swinunetr']
    assert gpu['objectives_checked'] == calibration['objectives_checked'] == 193
    assert gpu['real_training_cases_checked'] == calibration['training_cases'] == 10
    assert calibration['passed'] and calibration['warmup_sha256'] == sha(root/'shared_warmup/latest.pt')
    assert {e['id'] for e in p['runs']} <= set(gpu['objective_checks'])
    for name, expected in gpu['tested_sources'].items():
        assert sha(root/'source'/name) == expected, name
    for entry in p['runs']:
        assert sha(root/'calibration'/entry['id']/'record.json') == calibration['artifacts'][entry['id']]


def load(root):
    p = standard_protocol(verify_payload(root))
    scope = json.loads((root/SCOPE).read_text())
    assert scope['protocol_sha256'] == PROTOCOL_SHA and scope['excluded'] == list(EXCLUDED)
    assert scope['active_ids'] == [e['id'] for e in p['runs']]
    for name, expected in {**scope['added_sources'], **scope['evidence']}.items():
        assert sha(root/name) == expected, name
    validate_gates(root, p)
    return p, scope


def absent(root):
    for name in EXCLUDED:
        for path in (root/'runs'/name, root/'calibration'/name, root/'logs'/(name+'.log')):
            assert not path.exists() and not path.is_symlink(), path


def update_report(p, root):
    from .report import report
    from .retention import retain_checkpoints
    result = report(p, root)
    result['active_scope_sha256'] = sha(root/SCOPE)
    write(root/'results.json', result)
    retain_checkpoints(p, root, result)
    return result


def prepare(root):
    from .train import completed_record, identity
    from .warmup import prepare_prefix
    import torch
    queue = subprocess.check_output(['squeue', '-u', '3160552', '-h', '-o', '%i|%T|%j'], text=True)
    assert not queue.strip(), 'Wait for the cancelled allocation to exit: ' + queue
    with (root/'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        p = standard_protocol(verify_payload(root))
        prepare_prefix(p, root)
        validate_gates(root, p)
        if not (root/SCOPE).exists():
            preserved, resumes = {}, {}
            for entry in p['runs']:
                folder = root/'runs'/entry['id']
                if completed_record(folder, identity(p, entry)) is not None:
                    preserved[entry['id']] = sha(folder/'completion.json')
                elif (folder/'latest.pt').exists():
                    state = torch.load(folder/'latest.pt', map_location='cpu', weights_only=False)
                    assert state['identity'] == identity(p, entry)
                    assert state['scheduler']['step_size'] == 20
                    resumes[entry['id']] = dict(epoch=len(state['history']),
                        latest_sha256=sha(folder/'latest.pt'), best_sha256=sha(folder/'best.pt'))
            scope = dict(created_unix=time.time(), cancelled_job='694515', protocol_sha256=PROTOCOL_SHA,
                protocol_identity=p['identity_sha'], excluded=list(EXCLUDED), active_ids=[e['id'] for e in p['runs']],
                reason='User requested removal of longer, slower-decay experiments; preserve the original schedule.',
                completed_preserved=preserved, resumes=resumes,
                added_sources={n: sha(root/n) for n in ADDED},
                evidence={n: sha(root/n) for n in ('payload_manifest.json', 'cuda_preflight.json',
                    'calibration_preflight.json', 'shared_warmup/provenance.json')})
            write(root/SCOPE, scope)
        p, scope = load(root)
        # Only these exact nine paths are eligible for removal; reject every symlink.
        paths = [path for name in EXCLUDED for path in
                 (root/'runs'/name, root/'calibration'/name, root/'logs'/(name+'.log'))]
        receipt = root/'standard75_deletion.json'
        if not receipt.exists():
            records = []
            for path in paths:
                assert not path.is_symlink()
                files = list(path.rglob('*')) if path.is_dir() else [path]
                assert not any(f.is_symlink() for f in files)
                records.append(dict(path=str(path.relative_to(root)),
                    bytes=sum(f.stat().st_size for f in files if f.is_file())))
            write(receipt, dict(scope_sha256=sha(root/SCOPE), paths=records, state='prepared'))
        deletion = json.loads(receipt.read_text())
        assert deletion['scope_sha256'] == sha(root/SCOPE)
        for path in paths:
            assert not path.is_symlink()
            if path.is_dir():
                assert not any(f.is_symlink() for f in path.rglob('*'))
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
        absent(root)
        deletion.update(state='complete', completed_unix=time.time())
        write(receipt, deletion)
        result = update_report(p, root)
        write(root/'status.json', dict(state='interrupted_scope_amended', job_id='694515',
            completed_runs=result['completed'], planned_runs=190, current_stage=None,
            active_scope_sha256=sha(root/SCOPE), preserved_resumes=scope['resumes']))
        print(json.dumps(dict(completed=result['completed'], planned=190, resumes=scope['resumes'],
            deleted_bytes=sum(x['bytes'] for x in deletion['paths']), scope_sha256=sha(root/SCOPE))))


def submit(root):
    from .retention import quota_free
    with (root/'submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt, attempt = root/'standard75_submission.json', root/'standard75_submission_attempt.json'
        if receipt.exists():
            print(receipt.read_text()); return
        assert not attempt.exists(), 'Inspect unresolved submission attempt before retrying'
        p, scope = load(root)
        absent(root)
        assert json.loads((root/'standard75_deletion.json').read_text())['state'] == 'complete'
        assert not subprocess.check_output(['squeue', '-u', '3160552', '-h'], text=True).strip()
        free = quota_free()
        assert free >= p['minimum_free_gib']
        for name, resume in scope['resumes'].items():
            assert sha(root/'runs'/name/'latest.pt') == resume['latest_sha256']
        command = ['sbatch', '--parsable', f'--output={root}/slurm-%j.out', f'--error={root}/slurm-%j.err',
                   str(root/'source/scripts/band_focus/run_standard.sh'), str(root)]
        request = dict(command=command, requested_unix=time.time(), cancelled_job='694515',
            scope_sha256=sha(root/SCOPE), planned=190, free_gib=free, fold=0, seed=0, fraction='5pct',
            max_epochs=75, step_size=20, lr_gamma=.5, completed_preserved=len(scope['completed_preserved']))
        write(attempt, request)
        result = subprocess.run(command, cwd=root/'source', capture_output=True, text=True)
        request.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
        write(root/'standard75_submission_response.json', request)
        job = result.stdout.strip().split(';')[0]
        if result.returncode or not job.isdigit():
            raise RuntimeError('Submission unconfirmed; inspect response before retrying')
        request['job_id'] = job
        write(receipt, request)
        print(json.dumps(request))


def worker(root):
    import torch
    from .worker import verify
    from .warmup import prepare_prefix, seed_run
    from .train import completed_record, identity
    from .retention import quota_free
    torch.set_num_threads(4)
    with (root/'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = dict(job_id=os.environ['SLURM_JOB_ID'], state='verifying', stages=[],
                     started_unix=time.time(), planned_runs=190, active_scope_sha256=sha(root/SCOPE))
        write(root/'status.json', state)
        try:
            verify(root)  # Original source, dataset, dependencies and allocated bf16 GPU.
            p, scope = load(root)
            absent(root)
            prepare_prefix(p, root)
            for name, resume in scope['resumes'].items():
                assert sha(root/'runs'/name/'latest.pt') == resume['latest_sha256']
            result = update_report(p, root)
            state.update(state='running', completed_runs=result['completed'],
                         reused_verified_gpu_gate=True, calibrated_active_objectives=190)
            write(root/'status.json', state)
            for entry in p['runs']:
                if completed_record(root/'runs'/entry['id'], identity(p, entry)) is not None:
                    continue
                if quota_free() < 1.:
                    raise RuntimeError('Less than 1 GiB quota free; resume checkpoint preserved')
                seed_run(p, root, entry)
                state['current_stage'] = entry['id']
                write(root/'status.json', state)
                print('Starting ' + entry['id'], flush=True)
                started = time.monotonic()
                with (root/'logs'/(entry['id']+'.log')).open('a', buffering=1) as log:
                    child = subprocess.run([sys.executable, '-u', '-m', 'scripts.band_focus.standard_resume',
                        'train-one', '--root', str(root), '--run-id', entry['id']], cwd=root/'source',
                        stdout=log, stderr=subprocess.STDOUT)
                state['stages'].append(dict(stage=entry['id'], returncode=child.returncode,
                                           seconds=time.monotonic()-started))
                if child.returncode:
                    raise RuntimeError(entry['id'] + ' failed; inspect log')
                result = update_report(p, root)
                state['completed_runs'] = result['completed']
                write(root/'status.json', state)
            assert result['completed'] == 190
            state.update(state='complete', current_stage=None, finished_unix=time.time())
            write(root/'status.json', state)
        except Exception as error:
            state.update(state='failed', error=repr(error), finished_unix=time.time())
            write(root/'status.json', state)
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'submit', 'worker', 'train-one'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--run-id')
    args = parser.parse_args()
    if args.action == 'train-one':
        # Parent checked the payload and calibration; enforce the same scope here.
        from .train import run
        p = standard_protocol(json.loads((args.root/'protocol.json').read_text()))
        entry = next(e for e in p['runs'] if e['id'] == args.run_id)
        run(p, args.root, entry)
    else:
        globals()[args.action](args.root)


if __name__ == '__main__':
    main()
