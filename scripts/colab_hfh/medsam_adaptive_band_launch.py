"""Launch only the authorized bounded study after a safe previous-campaign pause."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import psutil

from scripts.colab_hfh.medsam_adaptive_band import CAMPAIGN, POLICY
from scripts.colab_hfh.medsam_adaptive_band_campaign import SOURCES, RUNS
from scripts.colab_hfh.medsam_constraints_drive import client
from scripts.colab_hfh.medsam_recovery import copy_verified
from scripts.colab_hfh.runtime import read, write, digest


def main():
    root = Path('/content/hippo_hfh'); old = root/'medsam_gain125_3fold3seed_v1'
    source = root/'bundle/source'; work = root/CAMPAIGN
    work.mkdir(exist_ok=True)
    state = read(old/'campaign_status.json')
    assert state['status'] == 'paused' and not state['workers'] and (old/'STOP').exists()
    for proc in psutil.process_iter(['cmdline']):
        if proc.pid == os.getpid() or proc.status() == psutil.STATUS_ZOMBIE:
            continue
        args = proc.info['cmdline'] or []
        assert not any(a.startswith('scripts.colab_hfh.medsam_') and
            any(k in a for k in ('worker','campaign','parallel')) for a in args), args
    assert not (work/'STOP').exists(), 'Explicit recovery audit required before resuming this experiment'
    os.environ.update(HFH_DRIVE_TRANSPORT='colab_adc', HFH_DRIVE_CREDENTIALS='gcloud')
    # Confirm durable old-run recovery without loading or changing its checkpoint.
    recovery = 'hfh_storage:results_medsam3_gain125_3fold3seed_v1/runs/gain125__dice__f2_s1/recovery'
    entries = json.loads(client().command('lsjson',recovery,'--files-only','--hash'))
    assert any(r['Name']=='latest.json' for r in entries)
    assert any(r['Name'].startswith('recovery-') and r['Size'] > 200_000_000 for r in entries)
    ledger = read(old/'runs/gain125__dice__f2_s1/checkpoint_ledger.json')
    plan = dict(created_at=time.time(),campaign=CAMPAIGN,policy=POLICY,runs=list(RUNS),
        source_sha256={n:digest(source/'scripts/colab_hfh'/n) for n in (*SOURCES,Path(__file__).name)},
        old_campaign_pause=dict(step=ledger['step'],protocol_sha256=ledger['protocol_sha256'],
            recovery_inventory=[dict(name=r['Name'],size=r['Size']) for r in entries]),
        maximum_active_workers=1,maximum_queued_jobs=1,baseline_reruns=0,
        stop_after_four=True,possible_revision='At most one evidence-justified revision; never automatically launched',
        user_checks='25%,50%,75%,100% of estimated wall time after verified start')
    write(work/'plan.json',plan)
    durable = Path('/content/hfh_storage/results_'+CAMPAIGN)
    copy_verified(work/'plan.json',durable/'plan.json')
    for name in (*SOURCES,Path(__file__).name):
        copy_verified(source/'scripts/colab_hfh'/name,durable/'source'/name)
    env = dict(os.environ,PYTHONPATH=str(source),CUBLAS_WORKSPACE_CONFIG=':4096:8',
               HFH_CONTROLLER_ARCHIVES='1',PYTHONUNBUFFERED='1')
    with (work/'controller.log').open('a') as log:
        process = subprocess.Popen([str(root/'venv/bin/python'),'-m',
            'scripts.colab_hfh.medsam_adaptive_band_campaign'],cwd=source,env=env,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(work/'launch_receipt.json',dict(**plan,controller_pid=process.pid,launched_at=time.time()))
    print('ADAPTIVE_LAUNCHED',json.dumps(dict(controller_pid=process.pid,work=str(work),runs=list(RUNS))),flush=True)


if __name__ == '__main__':
    main()
