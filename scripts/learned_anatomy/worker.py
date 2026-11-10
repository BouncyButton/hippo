"""Allocated sequence: prior fitting -> gradient gate -> preflight -> one run."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import torch
from scripts.depth_supervision.utils import atomic_json, digest


def verify(root):
    p = json.loads((root/'protocol.json').read_text())
    for name, expected in json.loads((root/'payload_manifest.json').read_text()).items():
        assert digest(root/name)==expected, name
    for name, expected in p['source_sha256'].items():
        assert digest(root/'source'/name)==expected, name
    for name, expected in json.loads((root/'roi_manifest.json').read_text())['files'].items():
        assert digest(root/'roi'/name)==expected
    assert digest(p['pkl'])==p['pkl_sha256']
    assert importlib.metadata.version('monai')=='1.6.0'
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    assert p['main_runs']==1 and len(p['splits']['5pct']['train'])==10
    return p


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--stage',choices=['frozen','encoder','refine','mass_refine','gate','preflight','swin'])
    a=parser.parse_args(); root=a.root
    torch.set_num_threads(4)
    p=verify(root)
    device=torch.device('cuda')
    if a.stage:
        if a.stage=='frozen':
            from .verify import verify_gate
            audit=verify_gate(root,p)
            record=json.loads((root/'encoder/mass_refinement_completion.json').read_text())
            assert record['encoder_sha256']==audit['encoder_sha256']
            assert record['identity']['protocol']==audit['protocol_sha256']
            assert record['train_cases']==p['splits']['5pct']['train'] and not record['validation_cases_used']
            atomic_json(root/'frozen_verification.json',dict(encoder_sha256=audit['encoder_sha256'],
                audit_passed=audit['passed'],exploratory_authorization=p['exploratory_audit_acceptance'],
                encoder_updates_this_job=0,verified=True))
        elif a.stage=='encoder':
            if p.get('reused_encoder'):
                reuse=p['reused_encoder']
                record=json.loads((root/'encoder/completion.json').read_text())
                initial=root/'encoder/initial_frozen.pt'
                assert digest(initial if initial.exists() else root/'encoder/frozen.pt')==reuse['sha256']==record['encoder_sha256']
                assert record['identity']==reuse['identity']
                assert record['identity']['train']==p['splits']['5pct']['train']
                assert record['identity']['settings']==p['encoder']
                assert record['steps']==p['encoder']['steps'] and not record['validation_cases_used']
                print('Verified and reused completed initial encoder fitting; no completed steps repeated.',flush=True)
                return
            from .prior import fit_prior
            fit_prior(root,p,device)
        elif a.stage=='refine':
            if p.get('reused_refinement'):
                record=json.loads((root/'encoder/refinement_completion.json').read_text())
                assert record==p['reused_refinement']
                before=root/'encoder/before_mass_refinement.pt'
                assert digest(before if before.exists() else root/'encoder/frozen.pt')==record['encoder_sha256']
                assert record['train_cases']==p['splits']['5pct']['train'] and not record['validation_cases_used']
                print('Verified and reused completed 300-step refinement; no completed steps repeated.',flush=True)
                return
            from .refine import run_refinement
            run_refinement(root,p,device)
        elif a.stage=='mass_refine':
            from .mass_refine import run_mass_refinement
            run_mass_refinement(root,p,device)
        elif a.stage=='gate':
            from .verify import run_gate
            run_gate(root,p,device)
        elif a.stage=='preflight':
            from .train import preflight
            preflight(root,p,device)
        else:
            from .train import run
            run(root,p,device)
        return
    (root/'logs').mkdir(exist_ok=True)
    state=dict(job_id=os.environ['SLURM_JOB_ID'],state='running',stages=[],started_unix=time.time(),protocol_sha256=p['identity_sha'])
    atomic_json(root/'environment.json',dict(torch=torch.__version__,monai=importlib.metadata.version('monai'),
        gpu=torch.cuda.get_device_name(),job_id=os.environ['SLURM_JOB_ID'],python=sys.version))
    try:
        stages = (['frozen','preflight','swin'] if p.get('exploratory_audit_acceptance') else
                  ['encoder']+(['refine'] if p.get('refinement') else [])+(['mass_refine'] if p.get('mass_refinement') else [])+['gate','preflight','swin'])
        for stage in stages:
            state['current_stage']=stage;atomic_json(root/'status.json',state)
            started=time.time()
            with (root/'logs'/(stage+'.log')).open('a',buffering=1) as log:
                ret=subprocess.run([sys.executable,'-u','-m','scripts.learned_anatomy.worker','--root',str(root),'--stage',stage],stdout=log,stderr=subprocess.STDOUT)
            state['stages'].append(dict(stage=stage,returncode=ret.returncode,seconds=time.time()-started))
            if ret.returncode:
                raise RuntimeError(stage+' failed; no subsequent training is allowed')
        done=json.loads((root/'runs/5pct_dice_learned_s0/completion.json').read_text())
        baseline=json.loads((root/'baseline_validation.json').read_text())
        atomic_json(root/'comparison.json',dict(baseline=baseline['summary'],learned=done['validation'],
            dice_delta_pp=100*(done['validation']['macro_dice']-baseline['summary']['macro_dice']),
            note='One fold, seed 0, 10 training cases, max 75 epochs; prior pretraining is additional compute.'))
        state.update(state='complete',current_stage=None,completed_main_runs=1,finished_unix=time.time())
    except Exception as error:
        state.update(state='failed',error=repr(error),finished_unix=time.time())
        raise
    finally:
        atomic_json(root/'status.json',state)


if __name__=='__main__':main()
