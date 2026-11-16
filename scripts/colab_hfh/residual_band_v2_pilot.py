"""Three fixed fold-zero seeds, then resume the other models; no expansion."""
import argparse
import fcntl
import hashlib
from pathlib import Path
import subprocess
import sys
import time
from scripts.colab_hfh.bands_campaign import read,write
from scripts.colab_hfh.constraint_drive import Drive
from scripts.colab_hfh.residual_band_v2_protocol import payload
from scripts.colab_hfh.nnunet_adaptive_pilot import pilot_summary

NAME='nnunet_conditional_band_v2'


def run(root,hours):
    work=root/NAME;work.mkdir(exist_ok=True)
    status=dict(status='waiting_for_current_matrix_objective',version=NAME,folds=[0],seeds=[0,1,2])
    path=work/'pilot_status.json';write(path,status)
    with (root/'bands_recipe_search.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (work/'STOP').exists():return
        (root/'constraint_search/STOP').unlink(missing_ok=True)
        deadline=time.time()+hours*3600
        plan=read(root/'constraint_matrix_plan.json')
        for name,expected in plan['calibration_implementation']['source_sha256'].items():
            if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()!=expected:
                raise ValueError('Common campaign source changed: '+name)
        frozen=payload(read(root/'bands_v1_payload.json'),plan)
        payload_path=work/'payload.json';write(payload_path,frozen)
        drive=Drive('results_'+NAME)
        def publish():write(path,status);drive.upload(path)
        try:
            status['status']='running_fold0';publish()
            print('ACTIVE-RAMP BAND: fold0 seeds0/1/2, same90updates and optimization cap; read-only probes separate',flush=True)
            subprocess.run([sys.executable,'-m','scripts.colab_hfh.residual_band_v2_campaign',
                '--root',str(root),'--payload',str(payload_path),'--work',str(work/'campaign'),
                '--folder','results_'+NAME+'_runs','--hours',str(max(.01,(deadline-time.time())/3600))],check=True)
            result=read(work/'campaign/campaign_status.json');status['results']=result['results']
            if result['status']!='complete_requested_matrix':
                status['status']='paused';publish();return
            status.update(status='complete_three_no_expansion',pilot=pilot_summary(result['results']))
            publish();print('V2 PILOT RESULT',status['pilot'],flush=True)
        except Exception as error:
            status.update(status='runtime_error',error=str(error));publish();raise
    if time.time()<deadline and not (root/'constraint_search/STOP').exists() and not (work/'STOP').exists():
        print('RESUMING THREE-MODEL MATRIX',flush=True)
        subprocess.run([sys.executable,'-m','scripts.colab_hfh.complete_selected_matrix',
            '--root',str(root),'--hours',str((deadline-time.time())/3600)],check=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--hours',type=float,default=8)
    a=p.parse_args();run(a.root,a.hours)
