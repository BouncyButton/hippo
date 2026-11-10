"""Freeze all 44 authorized conditions and their measured training schedules."""
import json
import shutil
from pathlib import Path

from campaign90 import BUDGET, MODELS, ROOT, digest, write_json
from scripts.capped_depth_delay.constants import ARMS, base_arm, CAMPAIGN
from scripts.capped_depth_delay.schedule import schedule_costs

def main():
    geometry_path = ROOT / 'docs/experiments/depth_swinunetr_20261006/protocol.json'
    geometry = json.loads(geometry_path.read_text())
    runs = []
    for identity, spec in MODELS.items():
        early = identity == 'nnunet'
        origin = 'nnunet_early_calibration_31976_20261006' if early else 'four_models_six_optimization_31976_20261006'
        parent_dir = ROOT / 'experiments' / origin / identity
        parent = json.loads((parent_dir / 'protocol.json').read_text())
        for name, sha in parent['training_source_sha256'].items():
            if digest(ROOT / name) != sha:
                raise ValueError('Parent source changed: ' + name)
        profile_path = ROOT / 'reports' / CAMPAIGN / 'profiles' / (identity + '.json')
        profile = json.loads(profile_path.read_text())
        for name, sha in profile['source_sha256'].items():
            if digest(ROOT / name) != sha:
                raise ValueError('Measured graph changed: ' + name)
        directory = ROOT / 'experiments' / CAMPAIGN / identity
        directory.mkdir(parents=True, exist_ok=True)
        for name in ('splits_10patients.json', 'nnunet_architecture.json', 'split_provenance.json'):
            shutil.copyfile(parent_dir / name, directory / name)
        split = json.loads((directory / 'splits_10patients.json').read_text())[0]
        if parent['pkl_sha256'] != geometry['pkl_sha256']:
            raise ValueError('Geometry and training data differ')
        native_geometry = {name: geometry['native_geometry'][name] for name in split['train'] + split['val']}
        write_json(directory / 'native_geometry.json', native_geometry)
        metadata = ROOT / 'reports' / origin
        if not early:
            metadata /= 'runs_metadata'
        references = {}
        comparison_endpoints = {}
        anchor = json.loads((metadata / identity / 'dice_bands/fold0/seed0/run/calibration.json').read_text())
        for arm in ARMS:
            original = base_arm(arm)
            path = metadata / identity / original / 'fold0/seed0/run/calibration.json'
            if path.exists() and original != 'dice':
                references[arm] = json.loads(path.read_text())
                comparison_endpoints[arm] = json.loads((path.parent / 'completion.json').read_text())['updates']
            else:
                references[arm] = {'anchor_model_sha256': anchor['anchor_model_sha256'], 'anchor_only': True}
        write_json(directory / 'calibration_references.json', references)
        costs = profile['costs']
        prefix, ramp = (9, 9) if early else (50, 50)
        schedules = {}
        for arm in ARMS:
            original = base_arm(arm)
            early_arm = arm.replace('_sdf', '')
            schedules[arm] = schedule_costs(costs['dice']['update_flops'], costs[early_arm]['update_flops'],
                costs[arm]['update_flops'], costs[arm]['calibration_flops'], prefix, ramp,
                has_sdf='sdf' in arm, has_depth='depth' in arm)
            schedule = schedules[arm]
            runs.append(dict(model=identity, trainer_alias=spec['alias'], objective=arm, fold=0, seed=0,
                             updates=schedule['max_steps'], epochs=schedule['patient_pass_epochs'],
                             calibration_update=prefix,
                             constraint_first_update=schedule['sdf_prefix_updates'] + 1 if arm == 'dice_sdf' else prefix + 1,
                             ramp_updates=ramp,
                             sdf_first_update=schedule['sdf_prefix_updates'] + 1 if 'sdf' in arm else None,
                             optimization_gflops=schedule['consumed_training_flops'] / 1e9,
                             calibration_gflops=schedule['calibration_flops'] / 1e9))
        sources = {**parent['training_source_sha256'], **profile['source_sha256']}
        sources.update({str(p.relative_to(ROOT)): digest(p)
                        for p in (ROOT / 'scripts/capped_depth_delay').glob('*') if p.is_file()})
        sources['scripts/depth_supervision/geometry.py'] = digest(ROOT / 'scripts/depth_supervision/geometry.py')
        protocol = {**parent, 'name': CAMPAIGN + '_' + identity,
                    'requested_arm_order': list(ARMS), 'training_source_sha256': sources,
                    'models': {spec['alias']: {'parameters': spec['parameters'], 'objectives': schedules}},
                    'constraint_warmup_updates': ramp,
                    'profile_evidence': {'file': str(profile_path.relative_to(ROOT)), 'sha256': digest(profile_path)},
                    'native_geometry': native_geometry, 'native_geometry_sha256': digest(directory / 'native_geometry.json'),
                    'depth_geometry': geometry['geometry'], 'depth_geometry_protocol_sha256': digest(geometry_path),
                    'calibration_references_sha256': digest(directory / 'calibration_references.json'),
                    'comparison_endpoints': comparison_endpoints,
                    'constraint_scale': 'Preserve existing early prefix/ramp. SDF starts at nearest update boundary to 5/9 of optimization cap, then uses the existing model ramp. Depth CE activates with other early terms.',
                    'calibration_policy': 'Ten unaugmented training volumes; frozen base coefficients reproduced at original anchor. Depth coefficient calibrated at that same anchor against the full native-plus-existing-auxiliary gradient. Calibration outside cap, separately reported.',
                    'depth_objective': 'Add original depth-weighted three-class cross-entropy. GT longitudinal PCA plane, physical 2-mm depth bins, 8-mm ROI; normalize bin masses with original sparse-bin rule. No class balancing/focal modulation. Recompute annotation weights with physical affine transformed consistently after augmentation.',
                    'sdf_objective': 'Original full-volume classwise SDF unchanged, independent of depth weights; only activation timing changes.',
                    'budget_scope': 'optimization_only',
                    'accounting': profile['accounting'],
                    'restart_policy': 'Fresh initialization per condition. Preserve all existing campaigns. Resume only identical frozen source, config, data, ledger and RNG.'}
        write_json(directory / 'protocol.json', protocol)
    if len(runs) != 44:
        raise ValueError('Exactly 12 delayed-SDF and 32 depth-containing runs required')
    plan = dict(name=CAMPAIGN, total_runs=44, delayed_sdf_runs=12, depth_runs=32,
                single_sbatch_allocation=True, sequential=True, fold=0, seed=0,
                budget_flops=BUDGET, budget_scope='optimization_only',
                calibration_in_cap=False, runs=runs,
                sdf_delay_fraction='5/9; nearest update boundary using actual pre-SDF optimization cost',
                depth_variant='original depth-weighted cross-entropy',
                storage='Node-local transient checkpoints; lossless best/final FP32 model states and verified metadata transferred to the local workspace before releasing each outbox export. Existing results preserved.')
    write_json(ROOT / 'experiments' / CAMPAIGN / 'plan.json', plan)
    write_json(ROOT / 'reports' / CAMPAIGN / 'prepared_plan.json', plan)
    print(json.dumps(plan), flush=True)


if __name__ == '__main__':
    main()
