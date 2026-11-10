"""One fail-fast GPU allocation; completed runs are always reused."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import torch
from scripts.depth_supervision.utils import atomic_json,digest
from scripts.unwrapped_swin.train import completed_record
from scripts.unwrapped_swin.worker import finish_stage
from .train import identity
from .report import report


def verify(root):
    p=json.loads((root/'protocol.json').read_text())
    for name,expected in json.loads((root/'payload_manifest.json').read_text()).items():
        if digest(root/name)!=expected:raise ValueError('Payload changed: '+name)
    for name,expected in p['source_sha256'].items():
        if digest(root/'source'/name)!=expected:raise ValueError('Source changed: '+name)
    refs=json.loads((root/'reference_manifest.json').read_text())
    for path,expected in refs['files'].items():
        if digest(path)!=expected:raise ValueError('Frozen reference changed: '+path)
    if digest(p['pkl'])!=p['pkl_sha256']:raise ValueError('Dataset changed')
    for package,version in p['required_dependencies'].items():
        if importlib.metadata.version(package)!=version:raise ValueError('Package version changed: '+package)
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Training requires allocated bf16 CUDA')
    atomic_json(root/'environment.json',dict(job_id=os.environ['SLURM_JOB_ID'],torch=torch.__version__,
        gpu=torch.cuda.get_device_name(),packages={k:importlib.metadata.version(k) for k in p['required_dependencies']},
        protocol_sha256=p['identity_sha']))
    return p


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root;torch.set_num_threads(4)
    (root/'logs').mkdir(exist_ok=True)
    state=dict(job_id=os.environ.get('SLURM_JOB_ID'),state='verifying',stages=[],started_unix=time.time())
    def execute(stage,argv):
        state['current_stage']=stage;atomic_json(root/'status.json',state)
        start=time.time();print('Starting '+stage,flush=True)
        with (root/'logs'/(stage+'.log')).open('a',buffering=1) as log:
            r=subprocess.run([sys.executable,'-u','-m',*argv],cwd=root/'source',stdout=log,stderr=subprocess.STDOUT)
        finish_stage(root,state,stage,r.returncode,time.time()-start)
    try:
        p=verify(root)
        state.update(protocol_sha256=p['identity_sha'],state='running')
        execute('followup_unit_checks',['scripts.unwrapped_followup.tests','--output',str(root/'unit_checks.json')])
        for fraction,arms in p['pairs'].items():
            execute(fraction+'_cuda_preflight',['scripts.unwrapped_followup.tests','--root',str(root),'--cuda-fraction',fraction])
            for arm in arms:
                execute(fraction+'_diagnostic_'+arm,['scripts.unwrapped_followup.diagnostics','--root',str(root),'--fraction',fraction,'--arm',arm])
            for entry in [e for e in p['runs'] if e['fraction']==fraction]:
                existing=completed_record(root/'runs'/entry['id'],identity(p,entry))
                if existing:
                    state['stages'].append(dict(stage=entry['id'],returncode=0,seconds=0,reused_completed=True))
                    atomic_json(root/'status.json',state)
                else:
                    execute(entry['id'],['scripts.unwrapped_followup.train','--root',str(root),'--run-id',entry['id']])
                report(p,root)
        result=report(p,root)
        if result['new_completed']!=p['main_runs']:raise RuntimeError('Missing completed runs')
        state.update(state='complete',current_stage=None,completed_main_runs=result['new_completed'],finished_unix=time.time())
        atomic_json(root/'status.json',state)
    except Exception as error:
        state.update(state='failed',error=repr(error),finished_unix=time.time())
        atomic_json(root/'status.json',state)
        raise


if __name__=='__main__':main()
