"""Cluster-only scheduled handoff; immutable bundle, no laptop dependency."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


STOP=False
ACTIVE=None


def stop(signum,frame):
    global STOP
    STOP=True
    if ACTIVE is not None and ACTIVE.poll() is None:ACTIVE.send_signal(signal.SIGUSR1)


def refine_until_handoff(bundle,spec):
    """Run registered frozen refinement stages serially within this allocation."""
    global ACTIVE
    attempted=set()
    for sig in (signal.SIGUSR1,signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
    while time.time()<spec['start_at']:
        control=json.loads((bundle/'control.json').read_text())
        if STOP or control['pause']:return False
        launch=None
        if time.time()<spec['start_at']-600:
            for entry in control.get('refinement_queue',[]):
                stage=Path(entry['stage'])
                if str(stage) in attempted or (stage/'bounded_completion.json').exists():continue
                assert stage.parent==Path(spec['parent']).parent and stage.name.startswith('mni_nnunet_swin_refine2030_round')
                assert hashlib.sha256((stage/'manifest.json').read_bytes()).hexdigest()==entry['manifest_sha256']
                preflight=json.loads((stage/'preflight.json').read_text());assert preflight['status']=='passed' and preflight['manifest_sha256']==entry['manifest_sha256']
                plan=json.loads((stage/'plan.json').read_text());assert plan['deadline']==spec['start_at'] and plan['gpu_workers']==2 and plan['budget_flops']==spec['budget_flops']
                assert entry['module'].startswith('scripts.mni_constraints.nnunet_swin_refine2030')
                if (stage/'failure.json').exists():attempted.add(str(stage));continue
                launch=entry;break
        if launch is None:
            time.sleep(min(10,max(0,spec['start_at']-time.time())));continue
        stage=Path(launch['stage']);attempted.add(str(stage))
        env={**os.environ,'PYTHONPATH':':'.join(str(stage/p) for p in ('source','source/scripts','source/baselines/nnUNet'))}
        with (stage/f"handoff-{os.environ['SLURM_JOB_ID']}.log").open('a') as log:
            ACTIVE=subprocess.Popen([spec['python'],'-u','-m',launch['module'],'run','--stage',str(stage)],cwd=stage/'source',env=env,stdout=log,stderr=subprocess.STDOUT)
            signaled=False
            while ACTIVE.poll() is None:
                control=json.loads((bundle/'control.json').read_text())
                if (STOP or control['pause'] or time.time()>=spec['start_at']-180) and not signaled:
                    ACTIVE.send_signal(signal.SIGUSR1);signaled=True
                time.sleep(3)
            (bundle/f"refinement_exit_{stage.name}.json").write_text(json.dumps(dict(stage=str(stage),exit_code=ACTIVE.returncode,finished_at=time.time())))
            ACTIVE=None
    return not STOP and not json.loads((bundle/'control.json').read_text())['pause']


def main():
    bundle=Path(sys.argv[1]);expected=sys.argv[2]
    assert hashlib.sha256((bundle/'manifest.json').read_bytes()).hexdigest()==expected
    manifest=json.loads((bundle/'manifest.json').read_text())
    for name,sha in manifest.items():assert hashlib.sha256((bundle/name).read_bytes()).hexdigest()==sha,name
    spec=json.loads((bundle/'spec.json').read_text())
    if not refine_until_handoff(bundle,spec):return
    loading=importlib.util.spec_from_file_location('winning_matrix_builder',bundle/'winning_matrix2030.py')
    module=importlib.util.module_from_spec(loading);loading.loader.exec_module(module)
    stage=Path(spec['stage']);plan=module.build(bundle,stage)
    env={**os.environ,'PYTHONPATH':':'.join(str(stage/p) for p in ('source','source/scripts','source/baselines/nnUNet'))}
    command=[spec['python'],'-u','-m',module.MODULE]
    subprocess.run(command+['preflight','--stage',str(stage)],cwd=stage/'source',env=env,check=True)
    receipt=dict(job_id=os.environ['SLURM_JOB_ID'],started_at=time.time(),scheduled_at=spec['start_at'],
        manifest_sha256=module.digest(stage/'manifest.json'),bundle_manifest_sha256=expected,
        reused=plan['reused_runs'],remaining=plan['num_runs'],laptop_required=False)
    module.write(stage/'submission.json',receipt);module.write(stage/'submissions'/f"{receipt['job_id']}.json",receipt)
    module.write(bundle/'handoff_completed.json',receipt)
    os.chdir(stage/'source')
    os.execve(spec['python'],command+['run','--stage',str(stage)],env)


if __name__=='__main__':main()
