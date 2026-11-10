"""Authorized 56-run MSD fold0/seed0 unaugmented Dice+CE target60 sweep."""
from copy import deepcopy
from pathlib import Path
import shutil
import subprocess
import sys

from scripts.cross_dataset_campaign import controller
from scripts.cross_dataset_campaign.common import ARMS, MODELS, CAP, read, write, digest
from scripts.cross_dataset_campaign.prepare import ROOT, freeze_source, schedule
from scripts.cross_dataset_campaign.probe_ce_bands_job import baseline_location

NAME = 'msd_four_models_ce_target60_20261014'
MODULE = 'scripts.cross_dataset_campaign.msd_ce_constraints_job'
ORIGINAL_COMMAND = controller.command_for


def prepare():
    stage = ROOT/'experiments'/NAME
    if (stage/'submission.json').exists():
        raise ValueError('Never change a submitted experiment')
    tasks = []
    from scripts.cross_dataset_campaign.storage import verify_archive
    for identity in MODELS:
        parent, reference_protocol, reference_run = baseline_location(identity)
        baseline = ROOT/'experiments'/parent/'protocols'/reference_protocol
        baseline_run = ROOT/'reports'/parent/'runs'/reference_run
        verify_archive(baseline_run, read(baseline_run/'archive_manifest.json'))
        historical = read(ROOT/'experiments/cross_dataset_full_20261014/protocols'/f'MSD_{identity}_f0_target10/protocol.json')
        profile_path = ROOT/'reports'/NAME/'profiles'/f'{identity}.json'
        profile = read(profile_path)
        if (profile['status'] != 'profiled' or profile['model'] != identity
                or profile['calibration_base'] != 'Dice+CE'
                or profile['objective_sha256'] != digest(ROOT/'scripts/cross_dataset_campaign/msd_ce_constraints.py')):
            raise ValueError('Exact current Dice+CE graph profile required')
        folder = stage/'protocols'/identity
        folder.mkdir(parents=True, exist_ok=True)
        for filename in ('splits_10patients.json','nnunet_architecture.json'):
            shutil.copyfile(baseline/filename, folder/filename)
        shutil.copyfile(profile_path, folder/'profile.json')
        p = deepcopy(read(baseline/'protocol.json'))
        p.pop('diagnostic_updates', None); p.pop('diagnostic', None)
        alias = p['trainer_alias']
        p.update(name=NAME, dataset='MSD', weight_ce=1, augmentation='none', folds=[0], seeds=[0],
            requested_arm_order=list(ARMS), constraints=True, train_count=10,
            training_loss='Established per-model Dice+CE, coefficients one, plus original calibrated auxiliaries',
            dice_include_background=identity != 'nnunet', calibration_target_ratio=.6,
            calibration_p95_cap=.75, calibration_base='Dice+CE', budget_flops=CAP, budget_gflops=CAP/1e9,
            sdf_policy=historical['sdf_policy'], depth_geometry=historical['depth_geometry'],
            constraint_warmup_updates=historical['constraint_warmup_updates'],
            validation_milestones=20, validation_target_flops=[CAP*i//20 for i in range(1,21)],
            checkpoint_selection='Final FLOP-budget checkpoint is primary; best validation is secondary',
            training_source_sha256={}, profile_sha256=digest(folder/'profile.json'),
            baseline_reference=dict(campaign=parent, run=reference_run,
                protocol_sha256=digest(baseline/'protocol.json'),
                initialization_sha256=read(baseline_run/'initialization.json')['model_state_sha256'],
                validation_sha256=digest(baseline_run/'validation_latest.json'),
                validation_dice=read(baseline_run/'validation_latest.json')['summary']['macro_hard_dice']))
        old = historical['models'][alias]['objectives']
        p['models'] = {alias:dict(parameters=profile['parameters'], objectives={
            arm:schedule(profile['costs'],arm,old[arm],10,CAP) for arm in ARMS})}
        write(folder/'protocol.json', p)
    # Compare all four models early, then proceed condition by condition.
    for arm in ARMS:
        for identity in MODELS:
            tasks.append(dict(id=f'MSD_{identity}_ce_{arm}_target60_f0s0', kind='conventional',
                dataset='MSD', model=identity, fold=0, seed=0, arm=arm, target=.6, protocol=identity))
    plan = deepcopy(read(ROOT/'experiments/msd_unet_ce_constraints_high_20261014/plan.json'))
    plan.update(name=NAME, num_runs=56, tasks=tasks, budget_flops_by_dataset={'MSD':CAP},
        archive_wait_timeout_seconds=300, archive_resume_module=MODULE,
        authorization='User requests immediate submission of four models, fold0 seed0, MSD 5%, all screenshot constraints at the discussed 60% target on Dice+CE without augmentation.',
        exploration_followup='Cross-dataset exploration and full three-fold/three-seed sweep remain held',
        comparison_policy='Compare each constraint with its fresh matched Dice+CE baseline; old augmented or Dice-only results are contextual, not matched 10% controls')
    plan.pop('baseline_reference_score', None)
    plan['storage'].update(archive=str(ROOT/'reports'/NAME), estimated_peak_logical_gib=5,
        mirrored_peak_gib=10, reserve_gib=2, minimum_free_gib=12)
    write(stage/'plan.json', plan)
    write(stage/'control.json', dict(pause=False, disabled_tasks=[], full_submission_enabled=False))
    write(stage/'source_version.json', dict(request=plan['authorization'],
        source_of_truth='Frozen source and SHA256 manifest include uncommitted implementation',
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
    freeze_source(stage)
    batch = (ROOT/'scripts/cross_dataset_campaign/run.sbatch').read_text().replace(
        'hippo-exploration-noaug','hippo-msd-four-ce60').replace(
        'scripts.cross_dataset_campaign.controller run', MODULE+' run')
    (stage/'run.sbatch').write_text(batch)
    write(stage/'manifest.json', {str(f.relative_to(stage)):digest(f) for f in sorted(stage.rglob('*'))
        if f.is_file() and f.name not in ('manifest.json','control.json')})
    print(stage)


def verify(stage):
    for name, sha in read(stage/'manifest.json').items():
        if digest(stage/name) != sha:
            raise ValueError('Frozen source changed: '+name)
    plan = read(stage/'plan.json')
    expected = [(m,a) for a in ARMS for m in MODELS]
    if (plan['name'] != NAME or plan['augmentation'] != 'none' or plan['num_runs'] != 56
            or [(t['model'],t['arm']) for t in plan['tasks']] != expected):
        raise ValueError('Expected the authorized four-by-fourteen matrix')
    for task in plan['tasks']:
        p = read(stage/'protocols'/task['protocol']/'protocol.json')
        profile = read(stage/'protocols'/task['protocol']/'profile.json')
        if ((task['fold'],task['seed'],task['target']) != (0,0,.6)
                or p['dataset'] != 'MSD' or p['augmentation'] != 'none' or p['weight_ce'] != 1
                or p['calibration_base'] != 'Dice+CE' or p['calibration_target_ratio'] != .6
                or p['calibration_p95_cap'] != .75 or p['budget_flops'] != CAP
                or p['model_identity'] != task['model']):
            raise ValueError('Unauthorized experiment setting')
        s = p['models'][p['trainer_alias']]['objectives'][task['arm']]
        if s != schedule(profile['costs'],task['arm'],s,10,CAP):
            raise ValueError('Schedule differs from measured FLOPs')
        if s['max_steps'] <= s['sdf_prefix_updates']:
            raise ValueError('No updates remain for a constraint')
    return plan


def command_for(stage, plan, task, base_path):
    command = ORIGINAL_COMMAND(stage, plan, task, base_path)
    command[2] = 'scripts.cross_dataset_campaign.msd_ce_constraints'
    return command


def preflight(stage):
    plan = verify(stage)
    from scripts.cross_dataset_campaign.preflight_msd_ce_constraints import check
    checks = check(stage, plan)
    subprocess.run(['bash','-n',str(stage/'run.sbatch')],check=True)
    write(stage/'preflight.json', dict(status='passed', manifest_sha256=digest(stage/'manifest.json'),checks=checks))
    print('All-four-models/all-fourteen-conditions Dice+CE preflight passed', flush=True)


def main():
    if len(sys.argv)>1 and sys.argv[1]=='prepare':
        prepare(); return
    controller.verify = verify
    controller.command_for = command_for
    if len(sys.argv)>1 and sys.argv[1]=='preflight':
        import argparse
        parser=argparse.ArgumentParser(); parser.add_argument('command'); parser.add_argument('--stage',type=Path,required=True)
        preflight(parser.parse_args().stage); return
    controller.main()


if __name__ == '__main__':
    main()
