"""Profile the exact 3D UNet Dice+CE constraint and calibration graphs."""
import argparse
from pathlib import Path

from scripts.cross_dataset_campaign import profile_msd_nnunet as profiler
from scripts.cross_dataset_campaign import probe_unet_constraints as objective
from scripts.cross_dataset_campaign.common import read, write, digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit = profiler.audit
    if audit.monai.__version__ != '1.6.0' or not audit.torch.__version__.startswith('2.2.2'):
        raise ValueError('Use the established profiling runtime')
    audit.torch.set_num_threads(2)
    audit.reference.supplementary_flops = profiler.intermediate_supplement
    profiler.dice_only_optimizer = objective.experiment.ORIGINAL_LOSS_OPTIMIZER
    profiler.training_loss = objective.training_loss
    profiler.calibration_case = objective.calibration_case
    origin = profiler.ROOT/'experiments/cross_dataset_full_20261014/protocols/MSD_basic_unet16_f0_target10'
    architecture, protocol = read(origin/'nnunet_architecture.json'), read(origin/'protocol.json')
    rows, costs = [], {}
    for arm in objective.ARMS:
        for calibration in ([False] if arm == 'dice' else [False, True]):
            measured = []
            for geometry in ('ellipsoid', 'box'):
                row = profiler.profile('basic_unet16', arm, geometry, calibration, architecture, protocol)
                rows.append(row); measured.append(row['flops'])
                print({k:row[k] for k in ('arm','geometry','calibration','flops')}, flush=True)
            if len(set(measured)) != 1:
                raise ValueError('Geometry changed the update/calibration cost')
            cost = costs.setdefault(arm, dict(calibration_flops=0))
            cost['calibration_flops' if calibration else 'update_flops'] = (
                10*measured[0]+(4096 if 'depth' in arm else 1024) if calibration else measured[0])
    reference = profiler.ROOT/'reports/msd_dice_ce_bands_pilot_20261014/profiles/basic_unet16.json'
    for arm in ('dice','dice_bands'):
        if costs[arm] != read(reference)['costs'][arm]:
            raise ValueError('Previously verified baseline/bands costs changed')
    write(args.output, dict(status='profiled', model='basic_unet16', costs=costs, profiles=rows,
        parameters=rows[0]['parameters'], input_shape=[1,1,64,64,64], calibration_base='Dice+CE',
        baseline_and_bands_reference_sha256=digest(reference),
        accounting='Established MAC=2 arithmetic; optimization cap; calibration measured separately',
        runtime=dict(torch=audit.torch.__version__, monai=audit.monai.__version__)))


if __name__ == '__main__':
    main()
