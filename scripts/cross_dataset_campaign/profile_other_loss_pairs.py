"""Profile unchanged Dice and Dice+CE graphs for the three other MSD models."""
import argparse
from functools import partial
from pathlib import Path

from scripts.cross_dataset_campaign import profile_msd_nnunet as profiler
from scripts.cross_dataset_campaign.common import read, write
from scripts.cross_dataset_campaign.probe_other_losses import paired_loss, MODELS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit = profiler.audit
    if audit.monai.__version__ != '1.6.0' or not audit.torch.__version__.startswith('2.2.2'):
        raise ValueError('Use the established local profiling runtime')
    audit.torch.set_num_threads(2)
    audit.reference.supplementary_flops = profiler.intermediate_supplement
    args.output.mkdir(parents=True, exist_ok=True)
    for identity in MODELS:
        origin = profiler.ROOT/'experiments/cross_dataset_full_20261014/protocols'/f'MSD_{identity}_f0_target10'
        architecture = read(origin/'nnunet_architecture.json')
        protocol = read(origin/'protocol.json')
        costs, rows = {}, []
        for weight_ce in (0, 1):
            profiler.training_loss = partial(paired_loss, weight_ce=weight_ce)
            arm_rows = []
            for geometry in ('ellipsoid', 'box'):
                row = profiler.profile(identity, 'dice', geometry, False, architecture, protocol)
                row['weight_ce'] = weight_ce
                rows.append(row)
                arm_rows.append(row)
                print({k:row[k] for k in ('model', 'weight_ce', 'geometry', 'flops')}, flush=True)
            measured = {r['flops'] for r in arm_rows}
            if len(measured) != 1:
                raise ValueError('Geometry changed the update cost')
            costs[str(weight_ce)] = measured.pop()
        write(args.output/f'{identity}.json', dict(status='profiled', model=identity,
            costs_by_ce_weight=costs, profiles=rows, parameters=rows[0]['parameters'],
            input_shape=[1, 1, 64, 64, 64],
            recipe='Existing MONAI Dice including background, unchanged AdamW; only CE toggled 0/1',
            runtime=dict(torch=audit.torch.__version__, monai=audit.monai.__version__)))


if __name__ == '__main__':
    main()
