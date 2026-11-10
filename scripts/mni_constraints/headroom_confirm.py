"""Confirm four training-headroom recipes across3folds×3seeds at one fixed cap."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil

from scripts.cross_dataset_campaign.common import read,write,digest,REMOTE,install_signals
from scripts.cross_dataset_campaign import controller
from scripts.cross_dataset_campaign.storage import verify_archive
from scripts.mni_constraints import budget_pilot as pilot, campaign
from scripts.unetr_subfields import cap_extension as batch
from scripts.unetr_subfields.contracts import subject

ROOT=campaign.ROOT
MODULE='scripts.mni_constraints.headroom_confirm'
NAME='mni_shared_cap3_headroom_3fold3seed_20261107'
SELECTION=ROOT/'reports/mni_shared_cap_headroom_20261107/selected_recipes.json'


def tasks(protocols):
    return [dict(id=f'MNI_{model}_f{fold}s{seed}',kind='conventional',dataset='MNI',model=model,
                 fold=fold,seed=seed,arm='dice',protocol=model,augmentation=protocols[model]['augmentation'])
            for seed in range(3) for fold in range(3) if (fold,seed)!=(0,0) for model in batch.MODELS]


def prepare():
    stage=ROOT/'experiments'/NAME
    if (stage/'manifest.json').exists():return verify(stage)
    selected=read(SELECTION)
    assert set(selected['models'])==set(batch.MODELS)
    assert selected['validation_floor']==.5
    assert selected['training_policy']=='flexible_headroom_no_rigid_dice_band'
    parent=ROOT/'experiments/mni_shared_cap3_headroom_refine_20261107'
    for part in ('source','data'):
        shutil.copytree(parent/part,stage/part,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(Path(__file__),stage/'source/scripts/mni_constraints/headroom_confirm.py')
    shutil.copy2(Path(__file__).with_name('summarize_headroom_confirm.py'),
                 stage/'source/scripts/mni_constraints/summarize_headroom_confirm.py')
    protocols={};reuse=[]
    for model,item in selected['models'].items():
        archive=ROOT/'reports'/item['stage']/'runs'/f'MNI_{model}_f0s0'
        verify_archive(archive,read(archive/'archive_manifest.json'))
        assert read(archive/'cleanup.json')['status']=='complete'
        train=read(archive/'training_final.json')['summary']['macro_hard_dice']
        validation=read(archive/'validation_latest.json')['summary']['macro_hard_dice']
        assert validation>selected['validation_floor'], (model,validation)
        assert 0<=train<=1
        assert digest(archive/'archive_manifest.json')==item['archive_manifest_sha256']
        origin=ROOT/'experiments'/item['stage']/'protocols'/model
        p=deepcopy(read(origin/'protocol.json'))
        config=read(archive/'config.json')
        assert config['protocol_sha256']==digest(origin/'protocol.json')
        assert config['model_identity']==model and config['fold']==config['seed']==0
        assert config['dataset_sha256']==p['pkl_sha256']
        assert config['augmentation']==p['augmentation']
        assert p['budget_flops']==selected['cap']
        folder=stage/'protocols'/model;shutil.copytree(origin,folder)
        p.update(name=NAME,data=str(REMOTE/NAME/'data/MNI.pkl'),folds=[0,1,2],seeds=[0,1,2],
                 learning_rate_policy=f"{p['lr_policy']}; cap-matched horizon={p['lr_horizon_updates']}; see lr_schedule_details for warmup and decay boundaries.",
                 selection_caveat='Recipes selected using training headroom, a50% validation floor and train/validation gap diagnostics atfold0seed0; confirm fixed recipes across3folds×3seeds. Validation is development data, not an untouched test.')
        write(folder/'protocol.json',p);protocols[model]=p
        reuse.append(dict(model=model,fold=0,seed=0,archive=str(archive),
            archive_manifest_sha256=item['archive_manifest_sha256'],training_dice=train,
            validation_dice=read(archive/'validation_latest.json')['summary']['macro_hard_dice'],
            original_protocol_sha256=digest(origin/'protocol.json')))
    plan=deepcopy(read(parent/'plan.json'))
    plan.update(name=NAME,tasks=tasks(protocols),num_runs=32,logical_runs=36,reused_runs=4,
        recipe_name='selected_model_recipes',
        lr_overrides={m:p['initial_lrs'][p['trainer_alias']] for m,p in protocols.items()},
        models=list(batch.MODELS),folds=[0,1,2],seeds=[0,1,2],archive_resume_module=MODULE,
        authorization='Continue the authorized MNI4baseline3fold3seed study at the lower shared cap. User clarified that85–90% training Dice is not a rigid requirement: validation must exceed50% and training should avoid overfitting. Reuse matchingfold0seed0 results; no constrained sweep.',
        criterion=dict(validation_dice_floor=.5,training_policy='flexible_headroom_no_rigid_dice_band',confirmation_report='Report all9cells per model: final training/validation Dice, gaps, late validation decline, saturation and subfield collapse diagnostics. Fixed shared cap; no per-run early stopping. Training Dice alone does not prove absence of overfitting.'),
        storage=dict(minimum_free_gib=12,estimated_peak_mirrored_gib=10,archive=str(ROOT/'reports'/NAME)))
    write(stage/'plan.json',plan);write(stage/'baseline_reuse.json',reuse)
    write(stage/'selection.json',selected);write(stage/'control.json',dict(pause=False))
    write(stage/'source_version.json',dict(parent_manifest_sha256=digest(parent/'manifest.json'),
        selection_sha256=digest(SELECTION),changes='Freeze selected per-model pilot recipes unchanged; expand to remaining8 fold/seed pairs per model and reuse4 pilot cells. Training/validation monitoring and per-run archive/retirement retained.'))
    (stage/'run.sbatch').write_text((parent/'run.sbatch').read_text().replace(pilot.MODULE,MODULE))
    campaign.freeze(stage)
    return verify(stage)


def verify(stage):
    for name,sha in read(stage/'manifest.json').items():assert digest(stage/name)==sha,name
    plan=read(stage/'plan.json');protocols={m:read(stage/'protocols'/m/'protocol.json') for m in batch.MODELS}
    assert plan['tasks']==tasks(protocols) and len(plan['tasks'])==32
    assert len(read(stage/'baseline_reuse.json'))==4 and plan['gpu_workers']==2
    for model,p in protocols.items():
        folder=stage/'protocols'/model;profile=read(folder/'profile.json')
        assert p['budget_flops']==plan['budget_flops']<campaign.CAP
        assert p['folds']==p['seeds']==[0,1,2] and not p['constraints']
        s=p['models'][p['trainer_alias']]['objectives']['dice']
        assert s==batch.schedule(profile['update_flops'],plan['budget_flops'])
        assert p['lr_horizon_updates']==s['max_steps']
        for name,key in [('profile.json','profile_sha256'),('splits_10patients.json','splits_sha256'),('nnunet_architecture.json','nnunet_architecture_sha256')]:assert digest(folder/name)==p[key]
        assert digest(stage/'data/MNI.pkl')==p['pkl_sha256']
        for split in read(folder/'splits_10patients.json'):
            assert len(split['train'])==len(split['val'])==10
            assert len({subject(c) for c in split['train']})==5
            assert not {subject(c) for c in split['train']}&{subject(c) for c in split['val']}
    return plan


def preflight(stage):
    plan=verify(stage)
    # Shapes/loss families are model-specific, not fold/seed-specific; verify all
    #32 identities above and exercise each of the4 architectures once below.
    representatives=[next(t for t in plan['tasks'] if t['model']==m) for m in batch.MODELS]
    original=pilot.verify
    try:
        unetr_protocol=read(stage/'protocols/unetr_vit_b16/protocol.json')
        pilot.verify=lambda _: {**plan,'tasks':representatives,'recipe_name':unetr_protocol['headroom_recipe']}
        result=pilot.preflight(stage)
    finally:pilot.verify=original
    result.update(verified_new_runs=32,reused_pilots=4)
    write(stage/'preflight.json',result);return result


def command_for(stage,plan,task,base_path=None):
    folder=stage/'protocols'/task['protocol'];p=read(folder/'protocol.json')
    return [plan['conventional_python'],'-u','-m',MODULE,'train','--model',p['trainer_alias'],
        '--objective','dice','--fold',str(task['fold']),'--seed',str(task['seed']),
        '--pkl',str(stage/'data/MNI.pkl'),'--protocol-dir',str(folder),
        '--output',str(stage/'runs'/task['id']),'--resume']


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('prepare')
    for command in ('verify','preflight','submit','run','run-one'):
        p=sub.add_parser(command);p.add_argument('--stage',type=Path,required=True)
        if command=='run-one':p.add_argument('--task',required=True)
    p=sub.add_parser('train');p.add_argument('--model',required=True);p.add_argument('--objective',required=True)
    for k in ('fold','seed'):p.add_argument('--'+k,type=int,required=True)
    for k in ('pkl','protocol-dir','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--threads',type=int,default=4)
    p.add_argument('--resume',action='store_true');p.add_argument('--smoke-steps',type=int,default=0)
    args=parser.parse_args();install_signals()
    controller.verify=verify;controller.command_for=command_for
    batch.verify=verify;batch.command_for=command_for;batch.RUNNER_MODULE=MODULE
    if args.command=='prepare':print(json.dumps(prepare()))
    elif args.command=='verify':print(json.dumps(verify(args.stage)))
    elif args.command=='preflight':print(json.dumps(preflight(args.stage)))
    elif args.command=='submit':print(json.dumps(controller.submit(args.stage)))
    elif args.command=='run':batch.run(args.stage)
    elif args.command=='run-one':batch.run_one(args)
    else:pilot.train(args)


if __name__=='__main__':main()
