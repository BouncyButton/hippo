"""Four paired bands-only MNI pilots using the unchanged lower-cap recipes."""
from copy import deepcopy
import json
import math
from pathlib import Path
import shutil

from scripts.cross_dataset_campaign.common import read, write, digest, REMOTE
from scripts.cross_dataset_campaign import controller
from scripts.cross_dataset_campaign.prepare import schedule
from scripts.cross_dataset_campaign.storage import verify_archive
from scripts.mni_constraints import campaign
from scripts.unetr_subfields import cap_extension as batch

ROOT=campaign.ROOT
NAME='mni_shared_cap3_bands_f0s0_20261107'
MODULE='scripts.mni_constraints.headroom_bands'
PARENT='mni_shared_cap3_headroom_3fold3seed_20261107'
TARGETS={'nnunet':.1,'swinunetr':.1,'unetr_vit_b16':.05,'basic_unet16':.1}
TRAIN=campaign.train
SUMMARIZE=controller.summarize


def tasks(protocols):
    return [dict(id=f'MNI_{m}_dice_bands_f0s0',kind='conventional',dataset='MNI',
                 model=m,arm='dice_bands',fold=0,seed=0,protocol=m,
                 augmentation=protocols[m]['augmentation'],target=TARGETS[m]) for m in batch.MODELS]


def band_schedule(profile,cap):
    costs=deepcopy(profile['costs'])
    costs['dice_bands']['update_flops']+=1  # Scalar coefficient product, as in the original study.
    ramp=max(1,math.ceil(.1*(cap//costs['dice']['update_flops'])))
    return schedule(costs,'dice_bands',dict(prefix_updates=ramp,
                    constraint_warmup_updates=ramp,sdf_prefix_updates=ramp+1),10,cap)


def prepare(_target=None):
    stage=ROOT/'experiments'/NAME;parent=ROOT/'experiments'/PARENT
    if (stage/'manifest.json').exists():return verify(stage)
    for part in ('source','data'):
        shutil.copytree(parent/part,stage/part,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(Path(__file__),stage/'source/scripts/mni_constraints/headroom_bands.py')
    parent_plan=read(parent/'plan.json');cap=parent_plan['budget_flops'];protocols={}
    reuse=read(parent/'baseline_reuse.json')
    for item in reuse:
        archive=Path(item['archive']);verify_archive(archive,read(archive/'archive_manifest.json'))
        assert digest(archive/'archive_manifest.json')==item['archive_manifest_sha256']
    evidence=read(ROOT/'reports/mni_constraint_target_search_f0s0_20261107/target_comparison.json')
    for model in batch.MODELS:
        observed=[r for r in evidence['results'] if r['model']==model and r['arm']=='dice_bands']
        assert max(observed,key=lambda r:(r['final_dice'],-r['target']))['target']==TARGETS[model]
        origin=parent/'protocols'/model;folder=stage/'protocols'/model
        shutil.copytree(origin,folder);p=deepcopy(read(origin/'protocol.json'))
        profile=read(ROOT/'reports/mni_constraints14_cap8_20261107/profiles'/f'{model}.json')
        assert profile['costs']['dice']['update_flops']==read(origin/'profile.json')['update_flops']
        assert profile['objective_sha256']==digest(stage/'source/scripts/mni_constraints/objectives.py')
        write(folder/'profile.json',profile);s=band_schedule(profile,cap);alias=p['trainer_alias']
        p.update(name=NAME,data=str(REMOTE/NAME/'data/MNI.pkl'),folds=[0],seeds=[0],
                 constraints=['bands'],requested_arm_order=['dice_bands'],
                 models={alias:dict(parameters=profile['parameters'],objectives={'dice_bands':s})},
                 profile_sha256=digest(folder/'profile.json'),lr_horizon_updates=s['max_steps'],
                 constraint_warmup_updates=s['constraint_warmup_updates'],
                 calibration_target_ratio=TARGETS[model],calibration_p95_cap=.75,
                 base_protocol_sha256=digest(origin/'protocol.json'),base_stage=PARENT,
                 constraint_recipe='Original two-voxel inner/outer foreground-union bands; no edge, SDF or depth term.',
                 timing_policy='Baseline-only first10% of baseline updates,10% ramp, then constant. Calibration uses training only and preserves RNG/model state.',
                 calibration_strength_policy='Fixed best observed bands-only target from the previous incomplete search; not a confirmed optimum. Recalibrate fresh against the current actual base loss;75% p95 guard.',
                 learning_rate_policy='Unchanged baseline peak and decay family; horizon matched to bands optimization-FLOP cost.',
                 lr_schedule_details=dict(policy=p['lr_policy'],horizon_updates=s['max_steps'],
                    warmup_updates=0 if model=='nnunet' else 24 if model=='unetr_vit_b16' else math.ceil(.05*s['max_steps']),
                    decay_start_index=s['max_steps']//2 if p['lr_policy']=='hold_half' else math.ceil(.05*s['max_steps'])))
        write(folder/'protocol.json',p);protocols[model]=p
    plan=deepcopy(parent_plan)
    plan.update(name=NAME,tasks=tasks(protocols),num_runs=4,logical_runs=8,reused_runs=4,
                folds=[0],seeds=[0],archive_resume_module=MODULE,calibration_targets=TARGETS,
                prerequisite_stage=str(REMOTE/PARENT),recipe_name='unchanged_headroom_recipes_plus_bands',
                criterion=dict(metric='Final-budget validation Dice difference against matchedfold0seed0 baseline; train/validation curves and subfield diagnostics secondary.'),
                storage=dict(minimum_free_gib=12,estimated_peak_mirrored_gib=10,archive=str(ROOT/'reports'/NAME)),
                authorization='User requested a queued singlefold/single seed bands test, leaving all baseline recipes unchanged and fixing nothing yet. Four models,fold0seed0,twoGPUworkers,one running and one queued allocation maximum.')
    write(stage/'plan.json',plan);write(stage/'baseline_reuse.json',reuse)
    write(stage/'target_selection_evidence.json',evidence);write(stage/'control.json',dict(pause=False))
    write(stage/'source_version.json',dict(parent_manifest_sha256=digest(parent/'manifest.json'),
          changes='Add only the existing bands objective and training-only calibration to the four frozen lower-cap recipes. Reuse existing constraint FLOP profiles with verified matching base costs/source. Match schedules to the same optimization cap; calibration remains separately accounted.'))
    (stage/'run.sbatch').write_text((parent/'run.sbatch').read_text().replace(
        'scripts.mni_constraints.headroom_confirm',MODULE).replace('hippo-mni-budget','hippo-mni-bands'))
    campaign.freeze(stage);return verify(stage)


def verify(stage):
    for name,sha in read(stage/'manifest.json').items():assert digest(stage/name)==sha,name
    plan=read(stage/'plan.json');protocols={m:read(stage/'protocols'/m/'protocol.json') for m in batch.MODELS}
    assert plan['tasks']==tasks(protocols) and plan['num_runs']==4 and plan['gpu_workers']==2
    assert len(read(stage/'baseline_reuse.json'))==4
    for m,p in protocols.items():
        folder=stage/'protocols'/m;profile=read(folder/'profile.json')
        s=p['models'][p['trainer_alias']]['objectives']['dice_bands']
        assert s==band_schedule(profile,plan['budget_flops'])
        assert p['budget_flops']==plan['budget_flops'] and p['lr_horizon_updates']==s['max_steps']
        assert campaign.training_flops(s['max_steps'],s)<=p['budget_flops']<campaign.training_flops(s['max_steps']+1,s)
        assert p['folds']==p['seeds']==[0] and p['constraints']==['bands']
        assert p['calibration_target_ratio']==TARGETS[m] and p['calibration_p95_cap']==.75
        assert digest(stage/'data/MNI.pkl')==p['pkl_sha256']
        for name,key in [('profile.json','profile_sha256'),('splits_10patients.json','splits_sha256'),('nnunet_architecture.json','nnunet_architecture_sha256')]:
            assert digest(folder/name)==p[key]
    return plan


def preflight(stage,protocol_names=None):
    verify(stage)
    import torch
    from scripts.mni_constraints import objectives as o
    from scripts.unetr_subfields import core,baseline_runtime
    from scripts.unetr_subfields.contracts import DATASETS
    from types import SimpleNamespace
    torch.set_num_threads(4);frame=core.base._load_pkl_dataframe(stage/'data/MNI.pkl');rows=[]
    for m in batch.MODELS:
        folder=stage/'protocols'/(protocol_names[m] if protocol_names else m);p=read(folder/'protocol.json');profile=read(folder/'profile.json')
        batch.bind(SimpleNamespace(protocol_dir=folder,model=p['trainer_alias'],objective='dice_bands',fold=0))
        model=core.base.build_model(p['trainer_alias'],read(folder/'nnunet_architecture.json')).to('meta')
        outputs=model(torch.zeros((1,1,64,64,64),device='meta'))
        shapes=[list(x.shape) for x in outputs] if isinstance(outputs,(list,tuple)) else [list(outputs.shape)]
        assert shapes==profile['output_shapes'] and sum(x.numel() for x in model.parameters())==profile['parameters']
        del model,outputs
        label=core.PreparedSubfields(frame,read(folder/'splits_10patients.json')[0]['train'],DATASETS['MNI'])[0]['label'][None]
        torch.manual_seed(0);outputs=[torch.randn(s,requires_grad=True) for s in shapes]
        if m!='nnunet':outputs=outputs[0]
        criterion=None
        if m=='nnunet':criterion,_=baseline_runtime.conventional.ORIGINAL_LOSS_OPTIMIZER(m,torch.nn.Linear(1,1),p)
        row=o.calibration_case(outputs,label,'dice_bands',criterion=criterion,identity=m)
        weights,stats=o.select_weights([row],'dice_bands',TARGETS[m],.75)
        loss=o.training_loss(outputs,label,'dice_bands',weights,1.,criterion=criterion,identity=m)
        logits=outputs[0] if isinstance(outputs,list) else outputs
        gradient=torch.autograd.grad(loss,logits)[0]
        assert torch.isfinite(loss) and torch.isfinite(gradient).all() and weights['bands']>0
        assert stats['bands']['weighted_median_ratio']<=TARGETS[m]+1e-8 and stats['bands']['weighted_p95_ratio']<=.75000001
        s=p['models'][p['trainer_alias']]['objectives']['dice_bands'];n=s['max_steps']
        assert batch.lr_at(n+10,n,p['initial_lrs'][p['trainer_alias']],m,p['lr_policy'])<=p['initial_lrs'][p['trainer_alias']]*.010001
        rows.append(dict(model=m,target=TARGETS[m],updates=n,prefix=s['prefix_updates'],ramp=s['constraint_warmup_updates'],calibration_flops=s['calibration_flops'],weights_for_synthetic_check_only=weights))
    result=dict(status='passed',manifest_sha256=digest(stage/'manifest.json'),checks=rows,optimizer_updates=0)
    write(stage/'preflight.json',result);return result


def train(args):
    from scripts.unetr_subfields import core
    p=read(args.protocol_dir/'protocol.json')
    def declared(protocol):
        assert protocol['augmentation']==p['augmentation'] and protocol['augmentation'] in ('none','xy_d4_v1')
    core.trainer.check_no_augmentation=declared
    TRAIN(args)


def summarize(stage,plan,status,active=None):
    SUMMARIZE(stage,plan,status,active)
    baseline={r['model']:r for r in read(stage/'baseline_reuse.json')};rows=[]
    for task in plan['tasks']:
        run=stage/'runs'/task['id']
        if not (run/'cleanup.json').exists():continue
        v=read(run/'validation_latest.json')['summary']['macro_hard_dice']
        t=read(run/'training_final.json')['summary']['macro_hard_dice'];b=baseline[task['model']]
        rows.append(dict(model=task['model'],target=task['target'],baseline_validation_dice=b['validation_dice'],
                         bands_validation_dice=v,delta_pp=100*(v-b['validation_dice']),
                         baseline_training_dice=b['training_dice'],bands_training_dice=t))
    state=read(stage/'status.json');state['bands_comparison']=rows;write(stage/'status.json',state)
    if status=='complete':write(stage/'bands_comparison.json',rows)


def run(stage):
    plan=verify(stage);parent=Path(plan['prerequisite_stage'])
    assert read(parent/'status.json')['status']=='complete','Baseline study must finish before bands pilot'
    batch.run(stage)


def main():
    campaign.MODULE=MODULE;campaign.prepare=prepare;campaign.verify=verify
    campaign.preflight=preflight;campaign.train=train;controller.summarize=summarize
    original_run=batch.run
    def checked_run(stage):
        plan=verify(stage)
        assert read(Path(plan['prerequisite_stage'])/'status.json')['status']=='complete','Baseline study must finish before bands pilot'
        original_run(stage)
    batch.run=checked_run
    campaign.main()


if __name__=='__main__':main()
