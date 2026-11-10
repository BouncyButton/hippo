"""Freeze the user-selected cap, measured schedules, and 24-run plan."""
import json
import shutil

from campaign90 import ARMS, BUDGET, CAMPAIGN, MODELS, ROOT, digest, validation_updates, write_json
from compute_constraint_objectives import scheduled_costs


def main():
    directory = ROOT / 'experiments' / CAMPAIGN
    reports = ROOT / 'reports' / CAMPAIGN
    origin = ROOT / 'experiments/nnunet_pair95_20261005/baseline'
    original = json.loads((origin/'protocol.json').read_text())
    reference_path = ROOT/'experiments/compute_budget_reference.json'
    snapshot = reports/'previous_compute_budget_reference.json'
    if not snapshot.exists():
        shutil.copyfile(reference_path, snapshot)
    paths = ['scripts/campaign90.py','scripts/train_campaign90.py','scripts/train_campaign90_core.py',
             'scripts/train_compute_matched_baselines.py','scripts/compute_constraint_objectives.py',
             'scripts/nnunet_multi_objective.py','scripts/nnunet_band_objective.py',
             'baselines/simple_segmentation.py','baselines/swin_unetr/swin_unetr.py',
             'thesis/new_constraints/training_augmentation.py',
             'thesis/new_constraints/bands/outer_boundary.py','thesis/new_constraints/edge_consistency.py',
             'thesis/new_constraints/signed_distance.py','thesis/new_constraints/constraint_result.py']
    source_hashes = {name: digest(ROOT/name) for name in paths}
    protocols, schedules = {}, {}
    targets = [BUDGET*i//20 for i in range(1,21)]
    for identity, spec in MODELS.items():
        profile_path = reports/'profiles'/f'{identity}.json'
        profile = json.loads(profile_path.read_text())
        for name, sha in profile['source_sha256'].items():
            if digest(ROOT/name) != sha:
                raise ValueError('Profile source changed: '+name)
        alias = spec['alias']
        objectives = {}
        for arm in ARMS:
            measured = profile['costs'][arm]
            schedule = scheduled_costs(BUDGET, spec['baseline_flops'], measured['update_flops'],
                                       measured['calibration_flops'], 50)
            schedule.update(flops_per_update=measured['update_flops'], baseline_flops_per_update=spec['baseline_flops'])
            objectives[arm] = schedule
        protocol = {key: original[key] for key in ['batch_size','spatial_size','num_classes','pkl_sha256',
                    'splits_sha256','nnunet_architecture_sha256','required_versions','profiled_versions',
                    'augmentation','precision','checkpoint_selection','adamw_weight_decay','sgd_weight_decay']}
        protocol.update(name=f'{CAMPAIGN}_{identity}', status='ready', model_identity=identity,
                trainer_alias=alias, folds=[0], seeds=[0], training_volumes_per_fold=10, validation_volumes_per_fold=52,
                budget_flops=BUDGET, budget_gflops=BUDGET/1e9, initial_lrs={alias:spec['initial_lr']},
                lr_horizon_updates=spec['lr_horizon'], nnunet_sgd_momentum=.95, nnunet_warmup_fraction=0.0,
                validation_milestones=20, validation_target_flops=targets, early_stopping=False,
                constraint_warmup_updates=50, requested_arm_order=list(ARMS),
                models={alias:{'parameters':spec['parameters'],'objectives':objectives}},
                calibration_policy='50 baseline updates, then ten unaugmented training volumes in eval mode. Detached FP32 logit gradients; median gradient ratio target0.1, p95 cap0.5; sequential bands, edge, SDF calibration. All calibration charged within each run cap.',
                constraint_scale='Ramp each calibrated auxiliary weight from1/50 to1 over50 constrained updates; full loss arithmetic charged during the ramp.',
                training_loss=('Native nnU-Net Dice+CE with unchanged deep supervision; constraints on full-resolution logits only' if alias=='nnunet' else 'Original background-including MONAI DiceLoss plus weighted constraints'),
                signed_distance_scope='Full-volume anterior/posterior target SDFs, clipped at8voxels and normalized; recomputed after augmentation. Target preprocessing excluded consistently.',
                edge_scope='Original pooled-foreground BoundaryEdgeConsistencyLoss; no separated edge variant.',
                learning_rate_policy=f'Preserve baseline LR trajectory with fixed {spec["lr_horizon"]}-update horizon, irrespective of objective or stopping cap. '+('SGD0.01, momentum0.95, polynomial exponent0.9, no warmup.' if alias=='nnunet' else f'AdamW{spec["initial_lr"]}, 5% horizon warmup, then cosine to1% initial LR.'),
                training_source_sha256=source_hashes,
                profile_evidence={'file':str(profile_path.relative_to(ROOT)),'sha256':digest(profile_path)},
                accounting=profile['accounting'], development_only=True,
                patient_identity_scope='Ten volume IDs, disjoint from52 validation volume IDs; original-MRI patient mapping unavailable.')
        model_dir = directory/identity
        model_dir.mkdir(parents=True,exist_ok=True)
        for name in ['splits_10patients.json','nnunet_architecture.json','split_provenance.json']:
            shutil.copyfile(origin/name,model_dir/name)
        write_json(model_dir/'protocol.json',protocol)
        protocols[identity] = digest(model_dir/'protocol.json')
        schedules[identity] = objectives
    plan = {'name':CAMPAIGN,'status':'ready','budget_flops_per_run':BUDGET,'budget_gflops_per_run':BUDGET/1e9,
            'fold':0,'seed':0,'model_order':list(MODELS),'arm_order':list(ARMS),'runs':24,
            'run_order':[[identity,arm] for arm in ARMS for identity in MODELS],
            'single_allocation':True,'protocol_sha256':protocols,'schedules':schedules,
            'pkl_sha256':original['pkl_sha256'],'training_volumes':10,'validation_volumes':52,
            'initial_lrs':{k:v['initial_lr'] for k,v in MODELS.items()},
            'lr_horizons':{k:v['lr_horizon'] for k,v in MODELS.items()},
            'budget_origin':'User selected90 times the measured native nnU-Net baseline training-update cost, following exploratory learning curves. No outcome filtering; all24 runs reported.',
            'development_only':True,'pretrained_weights':False,
            'prior_reference_sha256':digest(snapshot)}
    write_json(directory/'plan.json',plan)
    reference = {'status':'official_reference','adopted_on':'2026-10-06',
                 'budget_flops':BUDGET,'budget_gflops':BUDGET/1e9,
                 'definition':'90 native nnU-Net baseline updates;10 batch-one training volumes gives9 volume-pass epochs.',
                 'anchor_model':'nnunet','anchor_flops_per_update':MODELS['nnunet']['baseline_flops'],
                 'baseline_max_updates':90,'baseline_volume_pass_epochs':9.0,
                 'training_volumes':10,'validation_volumes':52,'batch_size':1,'input_shape':[1,1,64,64,64],
                 'selection_basis':plan['budget_origin'],'development_only':True,
                 'previous_reference_snapshot':str(snapshot.relative_to(ROOT)),
                 'previous_reference_sha256':digest(snapshot),'schedules_path':str((directory/'plan.json').relative_to(ROOT)),
                 'schedule_policy':'Keep each model baseline LR horizon fixed across every arm; stopping allowances are derived from measured objective and training-only calibration costs.',
                 'lr_horizons':plan['lr_horizons'],'accounting':profile['accounting']}
    write_json(reference_path,reference)
    write_json(reports/'prepared_plan.json',plan)
    rows = ['# Four-model comparison at31,976.45066709GFLOPs','',
            'One sequential allocation; fold0 seed0;10 training volumes,52 validation volumes;64³ batch1. All weights calibrated on training data inside the cap. Fixed baseline LR trajectories; fresh initialization. Development experiment, selected after exploratory results.','',
            '| Model | Objective | Updates | Epochs | Calibration GFLOPs | Consumed GFLOPs |',
            '|---|---|---:|---:|---:|---:|']
    for identity in MODELS:
        for arm,s in schedules[identity].items():
            rows.append(f'| {identity} | {arm} | {s["max_steps"]} | {s["patient_pass_epochs"]:.1f} | {s["calibration_flops"]/1e9:.6f} | {s["consumed_flops"]/1e9:.6f} |')
    (directory/'README.md').write_text('\n'.join(rows)+'\n')
    print(json.dumps({k:{a:s['max_steps'] for a,s in v.items()} for k,v in schedules.items()},indent=2))


if __name__=='__main__':
    main()
