"""Audit the exact Dice+CE/bands update and calibration graphs."""
import argparse
from pathlib import Path

from scripts.cross_dataset_campaign import profile_msd_nnunet as profiler
from scripts.cross_dataset_campaign import probe_ce_bands as objective
from scripts.cross_dataset_campaign.common import read, write, digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--models', nargs='+', choices=objective.MODELS, default=objective.MODELS)
    args = parser.parse_args()
    audit = profiler.audit
    if audit.monai.__version__ != '1.6.0' or not audit.torch.__version__.startswith('2.2.2'):
        raise ValueError('Use the established local profiling runtime')
    audit.torch.set_num_threads(2)
    audit.reference.supplementary_flops = profiler.intermediate_supplement
    profiler.dice_only_optimizer = objective.experiment.ORIGINAL_LOSS_OPTIMIZER
    profiler.training_loss = objective.training_loss
    profiler.calibration_case = objective.calibration_case
    for identity in args.models:
        origin = profiler.ROOT/'experiments/cross_dataset_full_20261014/protocols'/f'MSD_{identity}_f0_target10'
        reference = profiler.ROOT/'reports'/('msd_nnunet_loss_pair_90updates_20261014/profile.json'
            if identity == 'nnunet' else f'msd_other_models_loss_pairs_20261014/profiles/{identity}.json')
        reference_cost = read(reference)['costs_by_ce_weight']['1']
        architecture, protocol = read(origin/'nnunet_architecture.json'), read(origin/'protocol.json')
        rows, costs = [], {}
        for arm in ('dice', 'dice_bands'):
            for calibration in ([False] if arm == 'dice' else [False, True]):
                measured = []
                for geometry in ('ellipsoid', 'box'):
                    row = profiler.profile(identity, arm, geometry, calibration, architecture, protocol)
                    rows.append(row)
                    measured.append(row['flops'])
                    print({k:row[k] for k in ('model', 'arm', 'geometry', 'calibration', 'flops')}, flush=True)
                if len(set(measured)) != 1:
                    raise ValueError('Geometry changed the measured cost')
                cost = costs.setdefault(arm, dict(calibration_flops=0))
                cost['calibration_flops' if calibration else 'update_flops'] = 10*measured[0]+1024 if calibration else measured[0]
        if costs['dice']['update_flops'] != reference_cost:
            raise ValueError('Dice+CE baseline graph differs from the measured comparison')
        write(args.output/f'{identity}.json', dict(status='profiled', model=identity, costs=costs,
            profiles=rows, parameters=rows[0]['parameters'], input_shape=[1, 1, 64, 64, 64],
            baseline_profile_sha256=digest(reference), calibration_base='Dice+CE',
            accounting='Established dense arithmetic MAC=2; optimization cap; calibration separately reported',
            runtime=dict(torch=audit.torch.__version__, monai=audit.monai.__version__)))


if __name__ == '__main__':
    main()
