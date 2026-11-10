"""Deploy the next bounded refinement for the cluster-side 20:30 handoff."""
import json
import subprocess

from scripts.cross_dataset_campaign.common import read,write,digest,REMOTE
from scripts.launch_unetr_subfields import remote
from scripts.mni_constraints import nnunet_swin_refine2030_round3 as study
from scripts.mni_constraints.stage_seed5_remote import assemble
from scripts.mni_constraints.launch_unet_bands_timing_strength import collector


def main():
    stage=study.ROOT/'experiments'/study.NAME;report=study.ROOT/'reports'/study.NAME;dest=REMOTE/study.NAME
    study.verify(stage)
    parent=REMOTE/study.PARENT;env=':'.join(str(parent/p) for p in ('source','source/scripts','source/baselines/nnUNet'))
    code='''import json,sys,subprocess
from pathlib import Path
from scripts.cross_dataset_campaign.storage import quota
p=Path(sys.argv[1]);q=subprocess.check_output(['squeue','-h','-u','3160552','-o','%i|%T'],text=True).strip().splitlines()
own={'749648'};receipt=p.parent/'mni_matrix2030_handoff_bundle_20261109/timed_submission.json'
if receipt.exists():own.add(str(json.loads(receipt.read_text())['job_id']))
assert len(q)<=2 and all(r.split('|')[0] in own for r in q) and sum(r.split('|')[1]=='PENDING' for r in q)<=1,q
usage=quota();assert usage['free_gib']>=24
assert not (p/'submission.json').exists() and not (p/'status.json').exists();p.mkdir(exist_ok=True)
print(json.dumps(dict(quota=usage,queue=q)))'''
    write(report/'initial_cluster_check.json',json.loads(remote(['env','PYTHONPATH='+env,'python3','-c',code,dest])))
    print(json.dumps(assemble(stage,dest,report,read(stage/'deployment_known_donors.json'))),flush=True)
    env=':'.join(str(dest/p) for p in ('source','source/scripts','source/baselines/nnUNet'))
    result=json.loads(remote(['env','PYTHONPATH='+env,read(stage/'plan.json')['conventional_python'],'-m',study.MODULE,'preflight','--stage',dest],timeout=900).strip().splitlines()[-1])
    write(report/'remote_preflight.json',result)
    bundle=study.ROOT/'experiments/mni_matrix2030_handoff_bundle_20261109'
    control=read(bundle/'control.json');path=str(dest)
    if path not in control['refinement_stages']:control['refinement_stages'].append(path)
    entry=dict(stage=path,module=study.MODULE,manifest_sha256=digest(stage/'manifest.json'))
    control.setdefault('refinement_queue',[])
    if entry not in control['refinement_queue']:control['refinement_queue'].append(entry)
    write(bundle/'control.json',control)
    # The handoff queue is an explicit mutable control surface. Preserve a user
    # pause and existing entries when registering another verified study.
    code='''import json,sys,os
from pathlib import Path
p=Path(sys.argv[1]);entry=json.loads(sys.argv[2]);stage=Path(entry['stage'])
assert json.loads((stage/'preflight.json').read_text())['manifest_sha256']==entry['manifest_sha256']
value=json.loads((p/'control.json').read_text())
if entry['stage'] not in value['refinement_stages']:value['refinement_stages'].append(entry['stage'])
value.setdefault('refinement_queue',[])
if entry not in value['refinement_queue']:value['refinement_queue'].append(entry)
tmp=p/'control.register_tmp';tmp.write_text(json.dumps(value));os.replace(tmp,p/'control.json');print(json.dumps(value))'''
    merged=json.loads(remote(['python3','-c',code,REMOTE/bundle.name,json.dumps(entry)]))
    write(bundle/'control.json',merged)
    collector(dest,stage,report,'scripts.mni_constraints.collect_bounded_refinement')
    print(json.dumps(dict(status='passed',registered_for_handoff=entry)),flush=True)


if __name__=='__main__':
    try:main()
    except subprocess.CalledProcessError as error:print(error.output,error.stderr,flush=True);raise
