"""Archive and replace only the authorized held, never-started Slurm job."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from amend_views import atomic_json, held_state, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    args = parser.parse_args()
    root, candidate, job = args.root.resolve(), args.candidate.resolve(), '688228'
    if os.getuid() != 1395 or root.name != 'tissue_profiles_fold0_seed0_20261007_01':
        raise RuntimeError('Unexpected replacement target')
    with (root/'amendment.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        final = root/'replacement.json'
        if final.exists():
            print(final.read_text())
            return
        attempt = root/'replacement_attempt.json'
        if attempt.exists():
            raise RuntimeError('Unresolved replacement attempt; inspect preserved state and scheduler, do not resubmit blindly')
        before = held_state(job)
        if (root/'status.json').exists() or (root/'runs').exists():
            raise RuntimeError('Training artifacts exist; do not amend an experiment that started')
        original = json.loads((root/'protocol.json').read_text())
        updated = json.loads((candidate/'protocol.json').read_text())
        receipt = json.loads((root/'submission.json').read_text())
        assert receipt['job_id'] == job
        assert original['identity_sha'] == receipt['protocol_identity'] == updated['amendment']['previous_identity']
        assert original['main_runs'] == 12 and updated['main_runs'] == len(updated['runs']) == 36
        assert updated['runs'][:12] == original['runs']
        assert all(e['seed'] == 0 for e in updated['runs'])
        expected = [(f, 'informed', v) for f in ('5pct', 'full') for v in ('none', 'pooled', 'multiscale')]
        expected += [(f, a, 'none') for f, arms in updated['mri_view_pairs'].items() for a in arms]
        expected += [(f, a, v) for f, arms in updated['mri_view_pairs'].items() for a in arms for v in ('pooled', 'multiscale')]
        expected += [(f, a, v) for f, a in updated['native_mri_arms'].items() for v in ('none', 'pooled', 'multiscale')]
        assert [(e['fraction'], e['arm'], e['profile_variant']) for e in updated['runs'][12:]] == expected
        mutable = {'main_runs', 'runs', 'identity_sha', 'source_sha256', 'amendment', 'storage_budget', 'minimum_free_gib'}
        for key in set(original)-mutable:
            assert original[key] == updated[key], 'Existing experiment changed: '+key
        for deployment in (root, candidate):
            for name, expected_hash in json.loads((deployment/'payload_manifest.json').read_text()).items():
                if sha(deployment/name) != expected_hash:
                    raise ValueError('Payload changed: '+str(deployment/name))
        for name in ('profiles.py', 'multiscale.py', 'views.py', 'run.sh'):
            assert sha(root/'source/scripts/tissue_profiles'/name) == sha(candidate/'source/scripts/tissue_profiles'/name)
        checks = json.loads((candidate/'local_checks.json').read_text())
        assert checks['passed'] and checks['protocol_sha256'] == sha(candidate/'protocol.json')
        assert checks['unit']['passed'] and len(checks['informed']) == 3 and len(checks['mri_views']) == 18
        for path, expected_hash in json.loads((root/'reference_manifest.json').read_text())['files'].items():
            if sha(path) != expected_hash:
                raise ValueError('Frozen reference changed: '+path)
        assert sha(updated['pkl']) == updated['pkl_sha256']
        quota = subprocess.check_output(['beegfs', 'quota', 'list-usage', '--uids', 'current', '--pool', 'pool_home'], text=True)
        match = re.search(r'(\d+(?:\.\d+)?)GiB/(\d+(?:\.\d+)?)GiB', quota)
        if not match or float(match[2])-float(match[1]) < updated['minimum_free_gib']:
            raise RuntimeError('Insufficient or unreadable quota: '+quota)
        queue = subprocess.check_output(['squeue', '-u', '3160552', '-h', '-o', '%i|%T'], text=True).splitlines()
        for row in queue:
            jid, state = row.split('|')
            if not (jid == job and state == 'PENDING') and not (jid == updated['expected_predecessor'] and state == 'RUNNING'):
                raise RuntimeError('Unexpected queue state: '+repr(queue))
        held_state(job)
        backup = root/'pre_informed_deployment'
        backup.mkdir(exist_ok=False)
        for name in ('source', 'protocol.json', 'README.md', 'payload_manifest.json', 'payload.tar.gz',
                     'local_checks.json', 'submission.json', 'submission_attempt.json', 'submission_response.json',
                     'reference_manifest.json', 'amendment.json', 'amendment_views.json'):
            source = root/name
            if source.is_dir():
                shutil.copytree(source, backup/name)
            elif source.exists():
                shutil.copy2(source, backup/name)
        record = dict(replaced_job=job, state='installing', started_unix=time.time(), scheduler_before=before,
                      original_protocol_identity=original['identity_sha'], protocol_identity=updated['identity_sha'],
                      retained_runs=[e['id'] for e in original['runs']], added_runs=[e['id'] for e in updated['runs'][12:]],
                      runs=36, original_deployment=str(backup), quota=quota,
                      free_gib=float(match[2])-float(match[1]), cleanup_required=False)
        atomic_json(attempt, record)
        shutil.copytree(candidate/'source', root/'source', dirs_exist_ok=True)
        for name in ('protocol.json', 'README.md', 'payload_manifest.json', 'local_checks.json'):
            temporary = root/(name+'.installing')
            shutil.copy2(candidate/name, temporary)
            temporary.replace(root/name)
        shutil.copy2(candidate.parent/'informed_payload.tar.gz', root/'payload.tar.gz')
        for name, expected_hash in json.loads((root/'payload_manifest.json').read_text()).items():
            assert sha(root/name) == expected_hash
        held_state(job)
        record['state'] = 'installed_held'
        atomic_json(attempt, record)
        subprocess.run(['scancel', job], check=True)
        for _ in range(10):
            remaining = subprocess.check_output(['squeue', '-u', '3160552', '-h', '-o', '%i'], text=True).split()
            if job not in remaining:
                break
            time.sleep(.5)
        else:
            raise RuntimeError('Old job cancellation not yet confirmed; new submission not attempted')
        record['state'] = 'old_job_cancelled'
        atomic_json(attempt, record)
        # Exact old receipts were saved above; allow the existing idempotent
        # submitter to write a new attempt and receipt for the replacement job.
        for name in ('submission.json', 'submission_attempt.json', 'submission_response.json'):
            (root/name).unlink(missing_ok=True)
        result = subprocess.run([sys.executable, str(root/'source/scripts/tissue_profiles/submit.py')],
                                capture_output=True, text=True)
        record.update(submission_returncode=result.returncode, submission_stdout=result.stdout, submission_stderr=result.stderr)
        atomic_json(attempt, record)
        if result.returncode:
            raise RuntimeError('Replacement submission failed; inspect replacement_attempt.json: '+result.stderr)
        new = json.loads((root/'submission.json').read_text())
        assert new['job_id'] != job and new['protocol_identity'] == updated['identity_sha']
        record.update(state='replaced', job_id=new['job_id'], finished_unix=time.time(), dependencies=new['dependencies'])
        record['scheduler_after'] = subprocess.check_output(['scontrol', 'show', 'job', new['job_id'], '-o'], text=True)
        atomic_json(final, record)
        print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
