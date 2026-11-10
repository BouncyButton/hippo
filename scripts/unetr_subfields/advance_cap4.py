"""One-shot continuation of this specific MNI experiment; no recurring schedule."""
import json
from pathlib import Path
import subprocess
import time
import traceback
from scripts.cross_dataset_campaign.common import read,write
from scripts.unetr_subfields.contracts import ROOT
from scripts.unetr_subfields.report_cap_extension import summarize,finalize
from scripts.launch_unetr_subfields import remote

PHASES=[
    'mni_nnunet_cap8_full_20261106',
    'mni_swin_log_dice_cap8_pilot_20261106',
    'mni_swin_log_dice_cap8_full_20261106',
    'mni_unetr_geo_cap8_20261106',
    'mni_unet_cap8_20261106',
]


def main():
    output=ROOT/'reports/mni_baseline_rescue_20261106'
    try:
        for name in PHASES:
            report=ROOT/'reports'/name;last_count=-1
            write(output/'advance_status.json',dict(status='waiting_for_phase',phase=name,recurring=False))
            if not (report/'submission.json').exists():
                # Wait for the previous allocation to finish cancelling its own successor.
                while remote(['squeue','-h','-u','3160552','-o','%i']).strip():time.sleep(10)
                command=[str(ROOT/'.venv/bin/python'),'-u','-m','scripts.unetr_subfields.launch_cap_extension','--name',name,'--submit-only']
                subprocess.run(command,cwd=ROOT,check=True)
                print(json.dumps(dict(submitted=name)),flush=True)
            while True:
                if (report/'live.json').exists():
                    state=read(report/'live.json').get('status.json',{})
                    if state.get('status') in ('failed','paused'):
                        raise RuntimeError(name+': '+str(state.get('status')))
                if (report/'frozen/manifest.json').exists():
                    # Do not inspect an archive while the collector is still writing it.
                    count=sum(1 for f in (report/'runs').glob('*/cleanup.json') if read(f).get('status')=='complete')
                    if count!=last_count:
                        result=summarize(report);last_count=count
                        write(output/'advance_status.json',dict(status='running_phase',phase=name,completed=result['completed'],total=result['planned'],mean_dice=result['mean_dice'],recurring=False))
                        print(json.dumps(dict(phase=name,completed=result['completed'],total=result['planned'],mean_dice=result['mean_dice'])),flush=True)
                        if result['complete']:
                            passed=(result['minimum_dice']>=.5 and result['class_collapse_free'] if result['pilot'] else result['target_met'])
                            if not passed:
                                write(output/'advance_status.json',dict(status='needs_recipe_or_cap_adaptation',phase=name,result=result,recurring=False))
                                print(json.dumps(dict(needs_adaptation=name)),flush=True);return
                            print(json.dumps(dict(phase_passed=name,seed_means=result['seed_means'])),flush=True);break
                time.sleep(15)
        final=finalize([PHASES[i] for i in (0,2,3,4)],output)
        write(output/'advance_status.json',dict(status='complete',runs=final['completed_runs'],recurring=False))
        print(json.dumps(dict(status='all_four_baselines_complete',runs=36)),flush=True)
    except Exception:
        write(output/'advance_status.json',dict(status='controller_error',traceback=traceback.format_exc(),recurring=False))
        raise


if __name__=='__main__':main()
