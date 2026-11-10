"""Submit the verified width-48 baseline-only selection job once."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert os.getuid() == 1395
    stage = Path(__file__).resolve().parents[1]
    receipt = stage / 'submission.json'
    with (stage / 'submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if receipt.exists():
            print(receipt.read_text()); return
        for name, sha in json.loads((stage / 'payload_manifest.json').read_text()).items():
            assert digest(stage / name) == sha, name
        for name, sha in json.loads((stage / 'runtime_origin_manifest.json').read_text()).items():
            assert digest(stage / 'frozen_runtime' / name) == sha, name
        plan_path = stage / 'experiments/unet48_budget_selection_20261005/plan.json'
        plan = json.loads(plan_path.read_text())
        assert plan['target_min'] == .70 and plan['target_max'] == .75
        assert plan['initial_updates'] == 400 and plan['max_updates'] == 3200
        check = json.loads((stage / 'verification.json').read_text())
        assert check['cluster_tests_passed'] == 5 and check['plan_sha256'] == digest(plan_path)
        p = json.loads((plan_path.parent / 'template_protocol.json').read_text())
        assert digest(plan_path.parent / 'template_protocol.json') == plan['template_protocol_sha256']
        assert p['folds'] == p['seeds'] == [0] and p['models']['unet']['parameters'] == 43234202
        assert digest(Path('/mnt/beegfsstudents/home/3160552/hippo/datasets/Dataset101_MSD/msd_hippocampus_full.pkl')) == p['pkl_sha256']
        quota = subprocess.run(['beegfs','quota','list-usage','--uids','current','--pool','pool_home'],capture_output=True,text=True,check=True).stdout
        limits = re.search(r'(\d+(?:\.\d+)?)GiB/(\d+(?:\.\d+)?)GiB', quota)
        if not limits or float(limits[2]) - float(limits[1]) < 12:
            raise RuntimeError('Need at least 12 GiB free for pilot checkpoints: ' + quota)
        queue = subprocess.run(['squeue','-u','3160552','-h','-o','%i|%T|%j'],capture_output=True,text=True,check=True).stdout.splitlines()
        states = [r.split('|')[1] for r in queue if r.strip()]
        if states.count('PENDING') >= 1 or len(states) > 1:
            raise RuntimeError('One-running/one-queued limit: ' + repr(queue))
        sbatch = stage / 'scripts/run_unet48_budget_selection.sbatch'
        response = subprocess.run(['sbatch','--parsable',str(sbatch)],cwd=stage,capture_output=True,text=True,check=True)
        job = int(response.stdout.strip().split(';')[0])
        record = {'job_id':job,'submitted_at_unix':time.time(),'stage':str(stage),
                  'run_root':f'/mnt/beegfsstudents/home/3160552/compute_budget_unet48_{job}',
                  'plan_sha256':digest(plan_path),'sbatch_sha256':digest(sbatch),
                  'models_trained':['unet_width48'],'folds':[0],'seeds':[0],
                  'target_final_validation_dice':[.70,.75], 'maximum_pilots':10,
                  'queue_before_submission':queue,'quota_before_submission':quota}
        temporary = receipt.with_suffix('.json.tmp'); temporary.write_text(json.dumps(record,indent=2)+'\n'); temporary.replace(receipt)
        print(json.dumps(record,indent=2))


if __name__ == '__main__':
    main()
