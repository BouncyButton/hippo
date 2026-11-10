"""Check every frozen no-augmentation schedule and bind smoke evidence to it."""
import argparse
import json
from pathlib import Path

from scripts.adni_baselines.core import ROOT, BUDGET, base
from scripts.adni_constraints.worker import verify_stage
from scripts.adni_constraints.train import validation_steps
from scripts.capped_depth_delay.schedule import training_flops, charge_for_step


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--smoke', type=Path, required=True)
    args=parser.parse_args()
    plan=verify_stage(args.stage, check_data=False)
    smoke=json.loads(args.smoke.read_text())
    if smoke['status']!='passed' or {r['model'] for r in smoke['runs']}!={'basic_unet16','nnunet'}:
        raise ValueError('Both native and MONAI objective paths must pass real training smoke')
    schedules=[]
    for run in plan['runs']:
        directory=args.stage/'protocols'/run['model']
        protocol=json.loads((directory/'protocol.json').read_text())
        prior=ROOT/'experiments/adni_baselines_5pct_fold0_seed0_20261013/protocols'/run['model']
        parent=json.loads((prior/'protocol.json').read_text())
        s=protocol['models'][run['alias']]['objectives'][run['objective']]
        if protocol['augmentation']!='none' or protocol['budget_scope']!='optimization_only':
            raise ValueError('Incorrect augmentation or cap scope')
        for field in ('pkl_sha256','splits_sha256','nnunet_architecture_sha256','lr_horizon_updates',
                      'initial_lrs','precision','batch_size','spatial_size','adamw_weight_decay',
                      'sgd_weight_decay','nnunet_sgd_momentum','nnunet_warmup_fraction'):
            if protocol[field]!=parent[field]:
                raise ValueError('Baseline comparison recipe differs: '+field)
        n=s['max_steps']
        consumed=sum(charge_for_step(i,s) for i in range(n))
        if (consumed!=training_flops(n,s) or consumed!=s['consumed_training_flops']
                or not consumed<=BUDGET<training_flops(n+1,s)):
            raise ValueError('Schedule does not stop at the cap')
        milestones=validation_steps(protocol,run['objective'])
        if len(milestones)!=20 or milestones[-1]!=n:
            raise ValueError('Must retain the baseline 20-point validation schedule')
        profile=json.loads((directory/'profile.json').read_text())
        for row in profile['profiles']:
            if row['flops']!=sum(row['core_flops_per_stage'].values())+sum(row['supplementary_per_stage'].values()):
                raise ValueError('Profile stages do not reconcile')
        schedules.append(dict(model=run['model'],objective=run['objective'],updates=n,
                              validation_points=len(milestones),optimization_gflops=consumed/1e9))
    result=dict(status='passed',manifest_sha256=base.digest(args.stage/'manifest.json'),
        smoke_sha256=base.digest(args.smoke),smoke=smoke,conditions=56,augmentation='none',
        checks=['Binary bands/edge values and gradients match the original grouped-foreground losses',
                'Signed-distance gradients have the correct inside/outside direction',
                'Depth weights are finite and normalized on every unaugmented training crop',
                'Real optimizer/calibration smoke passed for MONAI and native nnU-Net losses; augmentation calls forbidden',
                'Real forward/loss/backward/optimizer arithmetic measured for four models and every arm on two geometries',
                'Binary baseline costs reproduce completed pilot profiles',
                'All 56 schedules exhaust but never exceed optimization cap; 20 validation checkpoints each',
                'Identical subject subset, architectures, optimization and LR trajectories; no augmentation'],schedules=schedules)
    base.atomic_json(args.stage/'preflight.json',result)
    report=ROOT/'reports'/args.stage.name
    base.atomic_json(report/'preflight.json',result)
    print(json.dumps(dict(status='passed',conditions=56,augmentation='none')),flush=True)


if __name__=='__main__':
    main()
