"""Submit only the six requested U-Net objectives at the selected fixed cap."""
import csv
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
        for n,s in json.loads((stage/'payload_manifest.json').read_text()).items():
            assert digest(stage/n)==s,n
        for n,s in json.loads((stage/'runtime_origin_manifest.json').read_text()).items():
            assert digest(stage/'frozen_runtime'/n)==s,n
        protocol_path=stage/'experiments/compute_matched_constraints_band_20261004/protocol.json'
        p=json.loads(protocol_path.read_text())
        objectives=['dice','dice_bands','dice_sdf','dice_bands_sdf','dice_bands_edge','dice_bands_edge_sdf']
        assert p['budget_flops']==12207124453303 and p['folds']==p['seeds']==[0]
        assert list(p['models'])==['unet'] and list(p['models']['unet']['objectives'])==objectives
        check=json.loads((stage/'verification.json').read_text())
        assert check['preflight_passed'] and check['protocol_sha256']==digest(protocol_path)
        pilot=Path('/mnt/beegfsstudents/home/3160552/compute_budget_band_682773/pilots/candidate00/run')
        selected=json.loads((stage/'experiments/compute_matched_constraints_band_20261004/budget_selection.json').read_text())
        assert digest(pilot/'completion.json')==selected['completion_sha256']
        done=json.loads((pilot/'completion.json').read_text())
        assert done['status']=='complete' and done['budget_flops']==p['budget_flops']
        with (pilot/'metrics.csv').open() as stream: final=list(csv.DictReader(stream))[-1]
        assert .70<=float(final['macro_hard_dice'])<=.77
        assert float(final['macro_hard_dice'])==selected['final_dice']
        retirement=Path('/mnt/beegfsstudents/home/3160552/compute_matched_constraints20_682721/RETIRED_UNET_20261004.json')
        assert retirement.exists() and not (retirement.parent/'unet').exists()
        data=Path('/mnt/beegfsstudents/home/3160552/hippo/datasets/Dataset101_MSD/msd_hippocampus_full.pkl')
        assert digest(data)==p['pkl_sha256']
        quota=subprocess.run(['beegfs','quota','list-usage','--uids','current','--pool','pool_home'],capture_output=True,text=True,check=True).stdout
        limits=re.search(r'(\d+(?:\.\d+)?)GiB/(\d+(?:\.\d+)?)GiB',quota)
        if not limits or float(limits[2])-float(limits[1])<5:raise RuntimeError('Insufficient free quota: '+quota)
        queue=subprocess.run(['squeue','-u','3160552','-h','-o','%i|%T|%j'],capture_output=True,text=True,check=True).stdout.splitlines()
        states=[line.split('|')[1] for line in queue if line.strip()]
        if states.count('PENDING')>=1 or len(states)>1:raise RuntimeError('One-running/one-queued limit: '+repr(queue))
        sbatch=stage/'scripts/run_constraints_band.sbatch'
        response=subprocess.run(['sbatch','--parsable',str(sbatch)],cwd=stage,capture_output=True,text=True,check=True)
        job=int(response.stdout.strip().split(';')[0])
        record={'job_id':job,'submitted_at_unix':time.time(),'stage':str(stage),
                'run_root':f'/mnt/beegfsstudents/home/3160552/compute_matched_constraints_band_{job}',
                'protocol_sha256':digest(protocol_path),'sbatch_sha256':digest(sbatch),
                'budget_flops_per_run':p['budget_flops'],'baseline_pilot_final_dice':selected['final_dice'],
                'models':['unet'],'folds':[0],'seeds':[0],'objectives':objectives,
                'queue_before_submission':queue,'quota_before_submission':quota}
        temporary=receipt.with_suffix('.json.tmp');temporary.write_text(json.dumps(record,indent=2)+'\n');temporary.replace(receipt)
        print(json.dumps(record,indent=2))


if __name__=='__main__':main()
