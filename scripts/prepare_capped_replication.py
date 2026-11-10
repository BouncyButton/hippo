"""Build a local, frozen 448-run bundle without uploading or submitting it."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile

sys.dont_write_bytecode = True

REPO = Path(__file__).resolve().parents[1]
PILOT = REPO / 'reports/capped_depth_delay_31976_retry_20261007'
FROZEN = PILOT / 'verified_metadata/source'
CAMPAIGN = 'capped_replication_31976_20261007'
STAGE = REPO / 'experiments' / CAMPAIGN
REPORT = REPO / 'reports' / CAMPAIGN
BUDGET = 31976450667090


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    STAGE.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    source = STAGE / 'source'
    shutil.copytree(FROZEN, source, dirs_exist_ok=True)
    for cache in source.rglob('__pycache__'):
        shutil.rmtree(cache)
    sys.path[:0] = [str(source), str(source / 'scripts')]
    from campaign90 import MODELS
    from scripts.capped_depth_delay.constants import ARMS
    from scripts.capped_depth_delay.schedule import training_flops, schedule_costs
    objectives = ('dice', 'dice_bands', 'dice_sdf', 'dice_bands_sdf', 'dice_bands_edge',
                  'dice_bands_edge_sdf', *(a for a in ARMS if 'depth' in a))
    assert len(objectives) == 14
    pilot = read(PILOT / 'summary.json') if (PILOT / 'summary.json').exists() else read(PILOT / 'verified_metadata/summary.json')
    old = read(REPO / 'reports/capped_constraint_results_31976_20261006/summary.json')
    references = read(PILOT / 'final_results_sources.json')
    original = old['runs']
    retained = references['retained_conditions']
    new = pilot['runs']
    decisions, reused, runs = [], [], []
    # Retain the validated geometry implementation; expand only case coverage.
    geometry_path = REPO / 'docs/experiments/depth_swinunetr_20261006/protocol.json'
    geometry = read(geometry_path)
    shutil.copyfile(geometry_path, source / 'replication_native_geometry_provenance.json')

    # Extend the production adapter through a separate file. Original training
    # modules remain byte-identical to the completed pilot.
    adapter = (source / 'scripts/capped_depth_delay/train.py').read_text()
    adapter = adapter.replace("    trainer.main()", """    trainer.ARMS = tuple(protocol['requested_arm_order'])
    original_loss = trainer.training_loss
    selected_policy = protocol['sdf_policy'][args.objective]
    if selected_policy == 'original_shared_ramp':
        from scripts.capped_depth_delay.objectives import full_loss
        def shared_ramp_loss(outputs, labels, arm, weights, scale, target_sdf=None,
                             criterion=None, depth_weights=None, sdf_scale=0.):
            if 'sdf' not in arm:
                return full_loss(outputs, labels, arm, weights, scale, criterion, target_sdf, depth_weights)
            # Keep the exact original shared ramp. The identity scalar multiply
            # plus the core's ramp division consume the two scheduled host FLOPs.
            adjusted = {**weights, 'sdf': weights['sdf'] * 1.0}
            return full_loss(outputs, labels, arm, adjusted, scale, criterion, target_sdf, depth_weights)
        trainer.training_loss = shared_ramp_loss
    else:
        trainer.training_loss = original_loss
    trainer.main()""")
    (source / 'scripts/capped_replication_train.py').write_text(adapter)
    verifier = (source / 'scripts/capped_depth_delay/verify.py').read_text()
    replacements = {
      'def verify_run(root, identity, arm, directory):': 'def verify_run(root, identity, arm, directory, fold=0, seed=0):',
      "output = root / identity / arm / 'fold0/seed0/run'": "output = root / identity / arm / f'fold{fold}/seed{seed}/run'",
      '(alias, arm, 0, 0)': '(alias, arm, fold, seed)',
      "json.loads((directory / 'splits_10patients.json').read_text())[0]": "json.loads((directory / 'splits_10patients.json').read_text())[fold]",
      "source / 'scripts/capped_depth_delay/train.py'": "source / 'scripts/capped_replication_train.py'",
      "'fold': 0, 'seed': 0": "'fold': fold, 'seed': seed",
    }
    for before, after in replacements.items():
        assert before in verifier, before
        verifier = verifier.replace(before, after)
    (source / 'scripts/capped_replication_verify.py').write_text(verifier)
    shutil.copyfile(REPO / 'scripts/capped_replication_worker.py', source / 'scripts/capped_replication_worker.py')

    protocols = {}
    for model, spec in MODELS.items():
        parent_dir = source / 'experiments/capped_depth_delay_31976_20261006' / model
        parent = read(parent_dir / 'protocol.json')
        assert geometry['pkl_sha256'] == parent['pkl_sha256']
        assert geometry['geometry'] == parent['depth_geometry']
        directory = source / 'experiments' / CAMPAIGN / model
        directory.mkdir(parents=True, exist_ok=True)
        for name in ('splits_10patients.json', 'nnunet_architecture.json', 'split_provenance.json'):
            shutil.copyfile(parent_dir / name, directory / name)
        splits = read(directory / 'splits_10patients.json')
        used = set(c for split in splits for c in split['train'] + split['val'])
        native_geometry = {c: geometry['native_geometry'][c] for c in sorted(used)}
        assert all(native_geometry[c] == parent['native_geometry'][c] for c in parent['native_geometry'])
        write(directory / 'native_geometry.json', native_geometry)
        profile = read(source / 'reports/capped_depth_delay_31976_20261006/profiles' / (model + '.json'))
        costs = profile['costs']
        prefix, ramp = (9, 9) if model == 'nnunet' else (50, 50)
        schedules, policies = {}, {}
        for arm in objectives:
            early_result = None
            late_result = next((r for r in new if r['model'] == model and r['objective'] == arm), None)
            if 'sdf' in arm and 'depth' not in arm:
                variant = 'early_calibration' if model == 'nnunet' else 'original'
                early_result = next(r for r in original if r['model'] == model and r['objective'] == arm
                                    and r['schedule_variant'] == variant)
                delayed = late_result['final_dice'] > early_result['final_dice']
                policy = 'delayed_5_of_9' if delayed else 'original_shared_ramp'
                reference = late_result if delayed else early_result
            elif 'sdf' in arm:
                policy = 'delayed_5_of_9'
                reference = late_result
            else:
                policy = 'not_applicable'
                reference = late_result or next(r for r in retained if r['model'] == model and r['objective'] == arm)
            row = copy.deepcopy(reference)
            row.update(fold=0, seed=0, reused=True, sdf_policy=policy,
                       pilot_selected=True, inference='Exploratory pilot; not an independent replication')
            reused.append(row)
            policies[arm] = policy
            if 'sdf' in arm:
                decisions.append(dict(model=model, objective=arm, selected_policy=policy,
                    original_dice_percent=100 * early_result['final_dice'] if early_result else None,
                    delayed_dice_percent=100 * late_result['final_dice'],
                    selection_basis='Higher final pilot Dice with identical calibration anchor' if early_result
                                     else 'Only tested timing for depth+SDF; retain approximately 56% delay'))
            if arm == 'dice':
                f = costs['dice']['update_flops']
                n = BUDGET // f
                schedule = dict(prefix_updates=0, sdf_prefix_updates=0,
                    baseline_flops_per_update=f, early_flops_per_update=f, flops_per_update=f,
                    calibration_flops=0, constraint_warmup_updates=ramp, sdf_warmup_updates=ramp,
                    budget_scope='optimization_only', sdf_delay_fraction_numerator=0,
                    sdf_delay_fraction_denominator=1, max_steps=n, consumed_training_flops=n*f,
                    consumed_flops=n*f, consumed_total_flops=n*f, unused_flops=BUDGET-n*f,
                    patient_pass_epochs=n/10)
            else:
                schedule = schedule_costs(costs['dice']['update_flops'],
                    costs[arm.replace('_sdf', '')]['update_flops'], costs[arm]['update_flops'],
                    costs[arm]['calibration_flops'], prefix, ramp, 'sdf' in arm, 'depth' in arm)
                if policy == 'original_shared_ramp':
                    schedule.update(sdf_prefix_updates=prefix,
                        early_flops_per_update=schedule['flops_per_update'],
                        sdf_delay_fraction_numerator=0, sdf_delay_fraction_denominator=1)
                    n = 0
                    while training_flops(n+1, schedule) <= BUDGET:
                        n += 1
                    consumed = training_flops(n, schedule)
                    schedule.update(max_steps=n, consumed_training_flops=consumed,
                        consumed_flops=consumed, consumed_total_flops=consumed+schedule['calibration_flops'],
                        unused_flops=BUDGET-consumed, patient_pass_epochs=n/10)
            schedules[arm] = schedule
        sources = dict(parent['training_source_sha256'])
        for name in ('capped_replication_train.py', 'capped_replication_worker.py', 'capped_replication_verify.py'):
            sources['scripts/' + name] = digest(source / 'scripts' / name)
        protocol = {**parent, 'name': CAMPAIGN + '_' + model, 'folds':[0,1,2], 'seeds':[0,1,2],
            'requested_arm_order':list(objectives),
            'models':{spec['alias']:{'parameters':spec['parameters'], 'objectives':schedules}},
            'native_geometry':native_geometry, 'native_geometry_sha256':digest(directory/'native_geometry.json'),
            'sdf_policy':policies, 'training_source_sha256':sources,
            'calibration_policy':'Recalibrate on the ten unaugmented training cases for this fold and seed, at update 9 for nnU-Net and 50 for other models. Calibration outside optimization cap; separately reported. No validation used.',
            'replication_policy':'Freeze pilot-selected timing before folds/seeds; preserve architectures, optimizer, precision, augmentation and learning-rate trajectories.',
            'restart_policy':'Skip only completed conditions with verified metadata. Fresh initialization for each missing condition.',
            'checkpoint_retention':'Verify full best/final checkpoints and preserve all JSON/CSV metadata before deleting completed scratch checkpoints.'}
        protocol.pop('calibration_references_sha256', None)
        write(directory / 'protocol.json', protocol)
        protocols[model] = protocol
    for fold in (0,1,2):
        for seed in (0,1,2):
            if fold == 0 and seed == 0:
                continue
            for model, spec in MODELS.items():
                for arm in objectives:
                    schedule = protocols[model]['models'][spec['alias']]['objectives'][arm]
                    runs.append(dict(model=model, trainer_alias=spec['alias'], objective=arm, fold=fold, seed=seed,
                        sdf_policy=protocols[model]['sdf_policy'][arm], updates=schedule['max_steps'],
                        epochs=schedule['patient_pass_epochs'],
                        optimization_gflops=schedule['consumed_training_flops']/1e9,
                        calibration_gflops=schedule['calibration_flops']/1e9))
    plan = dict(name=CAMPAIGN, models=list(MODELS), objectives=list(objectives), folds=[0,1,2], seeds=[0,1,2],
        total_new_runs=448, reused_conditions=56, total_conditions=504, runs=runs, reused_runs=reused,
        budget_flops=BUDGET, dataset_sha256=parent['pkl_sha256'], budget_scope='optimization_only',
        calibration_in_cap=False, single_sequential_job=True, checkpoint_retention='verified_metadata_only',
        selection_scope='Fold 0 / seed 0 timing selection is exploratory; new seeds/folds test a frozen policy.',
        estimated_hours_at_pilot_pace=(2*3600+13*60+47)/44*448/3600,
        walltime_hours=24, safe_stop_seconds=23*3600+40*60,
        automatic_submission=False, automatic_requeue=False,
        runtime_versions={'torch':'2.12.1+cu130','monai':'1.6.0'},
        parent_summary_sha256=digest(PILOT/'verified_metadata/summary.json'),
        retained_summary_sha256=digest(REPO/'reports/capped_constraint_results_31976_20261006/summary.json'))
    write(source / 'experiments' / CAMPAIGN / 'plan.json', plan)
    write(source / 'experiments' / CAMPAIGN / 'sdf_timing_selection.json', decisions)
    write(STAGE / 'plan.json', plan)
    write(STAGE / 'sdf_timing_selection.json', decisions)
    manifest = {str(p.relative_to(source)):digest(p) for p in sorted(source.rglob('*')) if p.is_file()}
    write(STAGE / 'source_manifest.json', manifest)
    sbatch = REPO / 'scripts/run_capped_replication_31976.sbatch'
    shutil.copyfile(sbatch, STAGE / sbatch.name)
    readme = '''Prepared only: this bundle has not been uploaded or submitted.

The single sbatch runs 448 remaining conditions sequentially: four models,
14 objectives, fold 0 seeds 1/2 and folds 1/2 seeds 0/1/2. It reuses 56
pilot-selected fold 0 / seed 0 conditions. All use the existing optimization
cap; calibration is excluded and measured separately. Calibration uses each
fold's own ten training patients. No new learning-rate tuning is performed.

Original SDF timing is retained for SwinUNETR and UNETR's three SDF objectives,
and for 3D U-Net bands+SDF. nnU-Net's three SDF objectives and 3D U-Net SDF-only
and bands+edge+SDF use the approximately 56% delay. Every depth+SDF objective
retains the tested approximately 56% delay, as no earlier-timing control exists.

All configurations, calibration, per-case validation, trajectories, best/final
scores and compute ledgers remain in results.md/results.csv and per-run JSON/CSV.
Full checkpoints are verified in node scratch, then discarded after metadata
storage is acknowledged. Later inference from those new checkpoints is unavailable.
Existing pilot checkpoints are not deleted by this job.

Slurm stud walltime is 24 hours. The worker stops between runs after 23h40m
and exits with paused_walltime if needed. Manually resubmitting this same sbatch
resumes only missing conditions. It never automatically submits/requeues.

When submission is authorized, upload the frozen bundle to:
/mnt/beegfsstudents/home/3160552/hippo/cluster_protocol_20261007/capped_replication_31976
Then run sbatch from that directory when the queue policy allows it.
Results persist at:
/mnt/beegfsstudents/home/3160552/capped_replication_31976_20261007_results
'''
    (STAGE/'README.md').write_text(readme)
    with tarfile.open(REPORT / 'submission_bundle.tar.gz', 'w:gz') as archive:
        for path in sorted(STAGE.rglob('*')):
            if path.is_file():
                archive.add(path, arcname=str(path.relative_to(STAGE)), recursive=False)
    write(REPORT/'preparation_receipt.json', dict(status='prepared_not_submitted', total_new_runs=448,
        total_reused_conditions=56, source_files=len(manifest),
        bundle_sha256=digest(REPORT/'submission_bundle.tar.gz'), sbatch_sha256=digest(STAGE/sbatch.name),
        plan_sha256=digest(source/'experiments'/CAMPAIGN/'plan.json'),
        source_manifest_sha256=digest(STAGE/'source_manifest.json'), checkpoint_retention='verified_metadata_only'))
    print(json.dumps({'status':'prepared_not_submitted', 'new_runs':len(runs), 'reused_conditions':len(reused),
        'bundle':str(REPORT/'submission_bundle.tar.gz'), 'estimated_hours':plan['estimated_hours_at_pilot_pace']}))


if __name__ == '__main__':
    main()
