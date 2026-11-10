"""Install a tested amendment only into the held, never-started authorized job."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def held_state(job):
    result = subprocess.check_output(['scontrol', 'show', 'job', job, '-o'], text=True).strip()
    if not all(part in result for part in ('JobState=PENDING ', 'Priority=0 ', 'RunTime=00:00:00 ')):
        raise RuntimeError('Amendment requires a held, never-started pending job: '+result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    args = parser.parse_args()
    root, candidate, job = args.root.resolve(), args.candidate.resolve(), '688228'
    if os.getuid() != 1395 or root.name != 'tissue_profiles_fold0_seed0_20261007_01':
        raise RuntimeError('Unexpected amendment target')
    with (root/'amendment.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root/'amendment.json').exists():
            print((root/'amendment.json').read_text())
            return
        before = held_state(job)
        if (root/'status.json').exists() or (root/'runs').exists():
            raise RuntimeError('Training artifacts exist; cannot change experiment identity')
        original = json.loads((root/'protocol.json').read_text())
        updated = json.loads((candidate/'protocol.json').read_text())
        receipt = json.loads((root/'submission.json').read_text())
        assert receipt['job_id'] == job
        assert original['identity_sha'] == receipt['protocol_identity'] == updated['amendment']['original_identity']
        assert updated['amendment']['job_id'] == job and updated['name'] == original['name']
        assert original['main_runs'] == 2 and updated['main_runs'] == len(updated['runs']) == 4
        expected = [(f, v) for f in ('5pct', 'full') for v in ('pooled', 'multiscale')]
        assert [(e['fraction'], e['profile_variant']) for e in updated['runs']] == expected
        assert all(e['seed'] == 0 for e in updated['runs'])
        for key in ('dataset', 'fold', 'seed', 'splits', 'native_geometry', 'pkl_sha256', 'model',
                    'feature_size', 'spatial_size', 'baseline_loss', 'learning_rate', 'weight_decay',
                    'step_size', 'lr_gamma', 'early_stopping', 'max_epochs', 'calibration', 'resources',
                    'profile_geometry', 'prior_root', 'atlas_protocol_sha256'):
            assert original[key] == updated[key], 'Original experiment changed: '+key
        for deployment in (root, candidate):
            for name, expected_hash in json.loads((deployment/'payload_manifest.json').read_text()).items():
                if sha(deployment/name) != expected_hash:
                    raise ValueError('Payload changed: '+str(deployment/name))
        # Keep the original scientific loss and the scheduler-captured script exact.
        for name in ('profiles.py', 'run.sh'):
            assert sha(root/'source/scripts/tissue_profiles'/name) == sha(candidate/'source/scripts/tissue_profiles'/name)
        checks = json.loads((candidate/'local_checks.json').read_text())
        assert checks['passed'] and checks['protocol_sha256'] == sha(candidate/'protocol.json')
        assert checks['multiscale']['passed'] and checks['unit']['cancellation_example']['pooled_loss'] == 0
        assert checks['unit']['cancellation_example']['multiscale_loss'] > 0
        for path, expected_hash in json.loads((root/'reference_manifest.json').read_text())['files'].items():
            if sha(path) != expected_hash:
                raise ValueError('Frozen reference changed: '+path)
        if sha(updated['pkl']) != updated['pkl_sha256']:
            raise ValueError('Dataset changed')
        quota = subprocess.check_output(['beegfs', 'quota', 'list-usage', '--uids', 'current', '--pool', 'pool_home'], text=True)
        match = re.search(r'(\d+(?:\.\d+)?)GiB/(\d+(?:\.\d+)?)GiB', quota)
        if not match or float(match[2])-float(match[1]) < updated['minimum_free_gib']:
            raise RuntimeError('Insufficient or unreadable quota: '+quota)
        held_state(job)
        backup = root/'original_deployment'
        backup.mkdir(exist_ok=False)
        # Preserve the entire original source and receipts before replacing files.
        preserved = ['source', 'protocol.json', 'README.md', 'payload_manifest.json', 'payload.tar.gz',
                     'local_checks.json', 'submission.json', 'submission_attempt.json', 'submission_response.json',
                     'reference_manifest.json']
        for name in preserved:
            source = root/name
            if source.is_dir():
                shutil.copytree(source, backup/name)
            else:
                shutil.copy2(source, backup/name)
        shutil.copytree(candidate/'source', root/'source', dirs_exist_ok=True)
        for name in ('protocol.json', 'README.md', 'payload_manifest.json', 'local_checks.json'):
            temporary = root/(name+'.installing')
            shutil.copy2(candidate/name, temporary)
            temporary.replace(root/name)
        shutil.copy2(candidate.parent/'amendment_payload.tar.gz', root/'payload.tar.gz')
        for name, expected_hash in json.loads((root/'payload_manifest.json').read_text()).items():
            assert sha(root/name) == expected_hash
        held_state(job)
        record = dict(job_id=job, state='installed_held', installed_at_unix=time.time(),
                      original_protocol_identity=original['identity_sha'], protocol_identity=updated['identity_sha'],
                      runs=4, retained_runs=[e['id'] for e in original['runs']],
                      added_runs=[e['id'] for e in updated['runs'] if e['profile_variant'] == 'multiscale'],
                      quota=quota, free_gib=float(match[2])-float(match[1]), cleanup_required=False,
                      scheduler_before=before, original_deployment=str(backup))
        receipt.update(protocol_sha256=sha(root/'protocol.json'), protocol_identity=updated['identity_sha'],
                       runs=4, objective='Dice + pooled or multiscale tissue profiles',
                       amendment=record, original_submission=str(backup/'submission.json'))
        atomic_json(root/'submission.json', receipt)
        atomic_json(root/'amendment.json', record)
        print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
