"""Freeze the three bands schedules from verified model-specific profiles."""
import copy
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compute_constraint_objectives import scheduled_costs
from run_budget_baselines import digest, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    original = ROOT / 'experiments/budget_baselines_94603_20261005'
    target = ROOT / 'experiments/budget_bands_94603_20261005'
    profiles = ROOT / 'reports/budget_bands_94603_20261005'
    campaign = copy.deepcopy(json.loads((original / 'campaign.json').read_text()))
    campaign.update(name='three_bands94603_fold0_seed0', baseline_job_id=684354, objective='dice_bands',
                    scope='Fresh baseline-plus-bands runs; one sequential Slurm job; native established model-specific losses and optimizers preserved.',
                    rounding='Literal user cap94,603.78GFLOPs; deduct the baseline prefix and calibration, then floor the remaining bands-update allowance.',
                    calibration_policy='50 baseline updates; one no-augmentation eval forward and detached-logit gradient calibration for each of the ten training volumes; target median bands/base gradient ratio0.1, p95 cap0.5; all calibration charged within each run cap.',
                    constraint_warmup_updates=50,
                    source_scope='Frozen baseline source plus the bands training extension; fresh training from scratch.')
    target.mkdir(parents=True, exist_ok=True)
    campaign['baseline_model_costs'] = campaign.pop('model_costs')
    campaign['baseline_max_updates'] = campaign.pop('max_updates')
    campaign['baseline_protocol_sha256'] = campaign.pop('protocol_sha256')
    campaign.update(model_costs={}, max_updates={}, protocol_sha256={}, calibration_flops={})
    for model in campaign['model_order']:
        source_dir = original / model
        directory = target / model
        directory.mkdir(exist_ok=True)
        for name in ['splits_10patients.json', 'split_provenance.json', 'nnunet_architecture.json']:
            shutil.copyfile(source_dir / name, directory / name)
        protocol = copy.deepcopy(json.loads((source_dir / 'protocol.json').read_text()))
        profile_path = profiles / (model + '_profiles.json')
        profile = json.loads(profile_path.read_text())
        for name, sha in profile['source_sha256'].items():
            if digest(ROOT / name) != sha:
                raise ValueError('Profile source changed: ' + name)
        entry = protocol['models'][model]
        if profile['parameters'] != entry['parameters']:
            raise ValueError('Profile architecture differs')
        costs = profile['costs']
        schedule = scheduled_costs(protocol['budget_flops'], costs['baseline'], costs['bands'], costs['calibration_total'], 50)
        schedule.update(flops_per_update=costs['bands'], baseline_flops_per_update=costs['baseline'])
        entry = {key: value for key, value in entry.items() if key in ['parameters', 'model_kwargs']}
        entry['objectives'] = {'dice_bands': schedule}
        protocol.update(name='budget94603_' + model + '_bands', models={model: entry},
                        requested_arm_order=['dice_bands'], constraint_warmup_updates=50,
                        baseline_job_id=684354, baseline_protocol_sha256=digest(source_dir / 'protocol.json'),
                        campaign_adapter_sha256=digest(ROOT / 'scripts/train_budget_bands.py'),
                        calibration_policy=campaign['calibration_policy'],
                        constraint_scale='After50 baseline updates, ramp from1/50 to1 over50 constrained updates; full bands graph charged throughout ramp.',
                        training_loss=('Native deep-supervised Dice+CE plus calibrated original bands on the full-resolution output' if model == 'nnunet' else 'Unchanged background-including Dice plus calibrated original bands'),
                        sources='Canonical two-step balanced BCE bands; gamma=0; degree_alpha=0; original band geometry; no edge or signed-distance term.',
                        pilot_scope='Exactly one bands run per model; fold0 seed0; fresh initialization matching baseline684354.',
                        profile_evidence={'profile_sha256': digest(profile_path), 'profile_file': str(profile_path.relative_to(ROOT)),
                                          'geometries': ['ellipsoid', 'box'], 'baseline_cost_verified_exact': True,
                                          'calibration_scope': profile['nnunet_calibration_scope'] if model == 'nnunet' else 'Gradients with respect to detached full-resolution Dice and bands logits'},
                        budget_accounting='50 baseline updates, all calibration attempts, all bands updates charged within94,603.78GFLOPs; no validation/data-preparation/I/O charge, matching the baseline convention.')
        write_json(directory / 'protocol.json', protocol)
        campaign['model_costs'][model] = costs['bands']
        campaign['max_updates'][model] = schedule['max_steps']
        campaign['calibration_flops'][model] = costs['calibration_total']
        campaign['protocol_sha256'][model] = digest(directory / 'protocol.json')
        print(json.dumps({'model': model, **schedule}), flush=True)
    write_json(target / 'campaign.json', campaign)


if __name__ == '__main__':
    main()
