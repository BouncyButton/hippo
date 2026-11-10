"""Authorized 56-run MSD sweep with model-specific baseline losses and targets."""
from copy import deepcopy
from pathlib import Path
import shutil
import sys

from scripts.cross_dataset_campaign import controller
from scripts.cross_dataset_campaign.common import ARMS, MODELS, CAP, read, write, digest
from scripts.cross_dataset_campaign.prepare import ROOT, freeze_source, schedule

NAME = 'msd_mixed_losses_nnunet60_others10_20261014'
PARENT = 'msd_four_models_ce_target60_20261014'
MODULE = 'scripts.cross_dataset_campaign.msd_mixed_losses_job'
BENCHMARK = 'msd_runtime_benchmark_v2_20261014'
ORIGINAL_COMMAND = controller.command_for


def prepare():
    from scripts.cross_dataset_campaign.storage import verify_archive
    parent = ROOT/'reports'/PARENT/'frozen'
    stage = ROOT/'experiments'/NAME
    if stage.exists():
        raise ValueError('Never overwrite an existing campaign')
    benchmark=ROOT/'reports'/BENCHMARK
    evidence=read(benchmark/'runs/summary.json');benchmark_manifest=read(benchmark/'manifest.json')
    if evidence['status']!='passed' or {r['model'] for r in evidence['models']}!=set(MODELS):
        raise ValueError('All four GPU equivalence benchmarks must pass before freezing')
    for row in evidence['models']:
        if not row['exact_loss_gradient_optimizer_model_calibration_metrics_rng_resume']:
            raise ValueError('Runtime optimization changed scientific results')
    for name,sha in benchmark_manifest['additional_sources'].items():
        if digest(ROOT/name)!=sha:raise ValueError('Optimization source changed after benchmark: '+name)
    acceleration=dict(enabled=True,gpu_data_cache=True,immutable_sdf_depth_cache=True,
        deferred_validation_readback=True,validation_batch_size=1,skip_unchanged_best_writes=True,
        benchmark=BENCHMARK,benchmark_summary_sha256=digest(benchmark/'runs/summary.json'),
        optimization_graph_precision_optimizer_and_resume_unchanged=True)
    for name,sha in read(parent/'manifest.json').items():
        if digest(parent/name)!=sha:
            raise ValueError('Parent source archive changed: '+name)
    stage.mkdir(parents=True)
    profile_sources = {}
    tasks = []
    for model in MODELS:
        native = model=='nnunet'
        target, ceiling = (.6,.75) if native else (.1,.5)
        folder = stage/'protocols'/model
        shutil.copytree(parent/'protocols'/model,folder)
        p = read(folder/'protocol.json')
        if native:
            reference = ROOT/'reports'/PARENT/'runs'/f'MSD_{model}_ce_dice_target60_f0s0'
            profile_origin = parent/'protocols'/model/'profile.json'
            profile = read(profile_origin)
            if profile['objective_sha256']!=digest(ROOT/'scripts/cross_dataset_campaign/msd_ce_constraints.py'):
                raise ValueError('Native profiled objective changed')
            source_check = {name:digest(parent/name) for name in read(parent/'manifest.json')
                if name.startswith('source/')}
            for name,sha in source_check.items():
                local = ROOT/str(Path(name).relative_to('source'))
                if local.exists() and digest(local)!=sha:
                    # Only report/management changes outside scientific source are
                    # acceptable; runtime source below is checked separately.
                    if any(token in name for token in ('capped_core.py','conventional.py','train_compute','objectives.py','nnunet_','baselines/','thesis/')):
                        raise ValueError('Native scientific source changed: '+name)
        else:
            reference = ROOT/'reports/msd_other_models_loss_pairs_20261014/runs'/f'MSD_{model}_dice_f0s0'
            profile_origin = ROOT/'experiments/cross_dataset_full_20261014/protocols'/f'MSD_{model}_f0_target10/profile.json'
            profile = read(profile_origin)
            for name,sha in profile['source_sha256'].items():
                if digest(ROOT/name)!=sha:
                    raise ValueError('Dice-only profiled source changed: '+name)
                profile_sources[name]=sha
            if digest(ROOT/profile['original_profile_file'])!=profile['original_profile_sha256']:
                raise ValueError('Original profile lineage changed')
        verify_archive(reference,read(reference/'archive_manifest.json'))
        shutil.copyfile(profile_origin,folder/'profile.json')
        old = p['models'][p['trainer_alias']]['objectives']
        p['models'][p['trainer_alias']]['objectives']={arm:schedule(profile['costs'],arm,old[arm],10,CAP) for arm in ARMS}
        p.update(name=NAME,weight_ce=int(native),calibration_base='Dice+CE' if native else 'Dice',
            runtime_acceleration=acceleration,
            calibration_target_ratio=target,calibration_p95_cap=ceiling,
            training_loss='Native foreground Dice+CE with established deep supervision' if native else 'Established MONAI Dice including background',
            profile_sha256=digest(folder/'profile.json'),baseline_reference=dict(
                campaign=reference.parents[1].name,run=reference.name,
                initialization_sha256=read(reference/'initialization.json')['model_state_sha256'],
                validation_sha256=digest(reference/'validation_latest.json'),
                validation_dice=read(reference/'validation_latest.json')['summary']['macro_hard_dice']))
        write(folder/'protocol.json',p)
    for arm in ARMS:
        for model in MODELS:
            native=model=='nnunet';target=.6 if native else .1
            tasks.append(dict(id=f'MSD_{model}_{"dicece" if native else "diceonly"}_{arm}_target{int(target*100)}_f0s0',
                kind='conventional',dataset='MSD',model=model,fold=0,seed=0,arm=arm,
                target=target,baseline_loss='Dice+CE' if native else 'Dice',protocol=model))
    plan = deepcopy(read(parent/'plan.json'))
    plan.update(name=NAME,tasks=tasks,num_runs=56,archive_resume_module=MODULE,
        authorization='User requested immediate MSD fold0 seed0 5% data job: all14 screenshot conditions, no augmentation; nnUNet native Dice+CE at60%, SwinUNETR/UNETR/3DUNet Dice-only at10%.',
        exploration_followup='ADNI/seismic/foundation and replicated sweeps remain held; prior MSD56 campaign remains abandoned',
        comparison_policy='Fresh matched baseline for each model; report constraint gains within each training recipe. This does not isolate architecture from baseline loss.',
        calibration_policy='Sequential targets: each auxiliary relative to the accumulated baseline and earlier auxiliaries; 60%/75% ceiling for nnUNet,10%/50% ceiling for others')
    plan['storage'].update(archive=str(ROOT/'reports'/NAME),estimated_peak_logical_gib=5,
        mirrored_peak_gib=10,reserve_gib=2,minimum_free_gib=12)
    write(stage/'plan.json',plan)
    shutil.copyfile(benchmark/'runs/summary.json',stage/'runtime_benchmark.json')
    shutil.copyfile(benchmark/'manifest.json',stage/'runtime_benchmark_manifest.json')
    write(stage/'control.json',dict(pause=False,disabled_tasks=[],full_submission_enabled=False))
    write(stage/'source_version.json',dict(parent_manifest_sha256=digest(parent/'manifest.json'),
        profile_policy='Exact existing per-objective measured profiles reused only after checking their source hashes and numerical loss/calibration identity',
        parent_campaign=PARENT,additional_profile_source_hashes=profile_sources))
    freeze_source(stage)
    for name,sha in profile_sources.items():
        dest=stage/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():shutil.copyfile(ROOT/name,dest)
        if digest(dest)!=sha:raise ValueError('Frozen profiled source differs: '+name)
    write(stage/'source/source_manifest.json',{str(f.relative_to(stage/'source')):digest(f)
        for f in sorted((stage/'source').rglob('*')) if f.is_file() and f.name!='source_manifest.json'})
    batch=(parent/'run.sbatch').read_text().replace('hippo-msd-four-ce60','hippo-msd-mixed-60-10').replace(
        'scripts.cross_dataset_campaign.msd_ce_constraints_job run',MODULE+' run')
    (stage/'run.sbatch').write_text(batch)
    write(stage/'manifest.json',{str(f.relative_to(stage)):digest(f) for f in sorted(stage.rglob('*'))
        if f.is_file() and f.name not in ('manifest.json','control.json')})
    print(stage)


def verify(stage):
    for name,sha in read(stage/'manifest.json').items():
        if digest(stage/name)!=sha:raise ValueError('Frozen source changed: '+name)
    plan=read(stage/'plan.json')
    if (stage/'runtime_benchmark.json').exists():
        evidence=read(stage/'runtime_benchmark.json')
        if evidence['status']!='passed' or {r['model'] for r in evidence['models']}!=set(MODELS):
            raise ValueError('Runtime benchmark incomplete')
        for name,sha in read(stage/'runtime_benchmark_manifest.json')['additional_sources'].items():
            if digest(stage/'source'/name)!=sha:raise ValueError('Runtime source differs from benchmark')
    if (plan['name']!=NAME or plan['augmentation']!='none' or plan['num_runs']!=56
            or [(t['model'],t['arm']) for t in plan['tasks']]!=[(m,a) for a in ARMS for m in MODELS]):
        raise ValueError('Expected exactly four models by14 conditions')
    for task in plan['tasks']:
        folder=stage/'protocols'/task['protocol'];p=read(folder/'protocol.json')
        acceleration=p.get('runtime_acceleration',{})
        if acceleration.get('enabled'):
            if (acceleration['validation_batch_size']!=1 or not (stage/'runtime_benchmark.json').exists()
                    or acceleration['benchmark_summary_sha256']!=digest(stage/'runtime_benchmark.json')):
                raise ValueError('Runtime acceleration is not bound to equivalence evidence')
        native=task['model']=='nnunet';target,ceiling=(.6,.75) if native else (.1,.5)
        if ((task['fold'],task['seed'],task['target'])!=(0,0,target)
                or p['dataset']!='MSD' or p['model_identity']!=task['model']
                or p['augmentation']!='none' or p['weight_ce']!=int(native)
                or p['calibration_base']!=('Dice+CE' if native else 'Dice')
                or p['calibration_target_ratio']!=target or p['calibration_p95_cap']!=ceiling
                or p['budget_flops']!=CAP):
            raise ValueError('Unauthorized experiment settings')
        for filename,key in [('splits_10patients.json','splits_sha256'),('nnunet_architecture.json','nnunet_architecture_sha256'),('profile.json','profile_sha256')]:
            if digest(folder/filename)!=p[key]:raise ValueError('Artifact changed: '+filename)
        profile=read(folder/'profile.json')
        if native:
            if profile['objective_sha256']!=digest(stage/'source/scripts/cross_dataset_campaign/msd_ce_constraints.py'):
                raise ValueError('Native objective profile changed')
        else:
            for name,sha in profile['source_sha256'].items():
                if digest(stage/'source'/name)!=sha:raise ValueError('Dice profile source changed: '+name)
        s=p['models'][p['trainer_alias']]['objectives'][task['arm']]
        if s!=schedule(profile['costs'],task['arm'],s,10,CAP):raise ValueError('Measured FLOP schedule differs')
        if s['max_steps']<=s['sdf_prefix_updates']:raise ValueError('No updates remain for constraints')
    return plan


def command_for(stage,plan,task,base_path):
    command=ORIGINAL_COMMAND(stage,plan,task,base_path)
    command[2]='scripts.cross_dataset_campaign.msd_mixed_losses'
    return command


def main():
    if len(sys.argv)>1 and sys.argv[1]=='prepare':prepare();return
    controller.verify=verify;controller.command_for=command_for
    if len(sys.argv)>1 and sys.argv[1]=='preflight':
        import argparse
        from scripts.cross_dataset_campaign.preflight_msd_mixed_losses import check
        parser=argparse.ArgumentParser();parser.add_argument('command');parser.add_argument('--stage',type=Path,required=True)
        stage=parser.parse_args().stage;checks=check(stage,verify(stage))
        write(stage/'preflight.json',dict(status='passed',manifest_sha256=digest(stage/'manifest.json'),checks=checks))
        print('MSD mixed-loss56-run preflight passed',flush=True);return
    controller.main()


if __name__=='__main__':main()
