"""Measure both sides of the native Dice versus Dice+CE ablation."""
import argparse
import json
from pathlib import Path

from scripts.cross_dataset_campaign import profile_msd_nnunet as profiler
from scripts.cross_dataset_campaign.common import write
from scripts.cross_dataset_campaign.conventional import ORIGINAL_LOSS_OPTIMIZER


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit = profiler.audit
    if audit.monai.__version__ != '1.6.0' or not audit.torch.__version__.startswith('2.2.2'):
        raise ValueError('Use the established local profiling runtime')
    audit.torch.set_num_threads(2)
    audit.reference.supplementary_flops = profiler.intermediate_supplement
    origin = profiler.ROOT / 'experiments/nnunet_pair95_20261005/baseline'
    architecture = json.loads((origin / 'nnunet_architecture.json').read_text())
    protocol = json.loads((origin / 'protocol.json').read_text())
    protocol['depth_geometry'] = json.loads((profiler.ROOT / 'docs/experiments/depth_swinunetr_20261006/protocol.json').read_text())['geometry']
    costs, rows = {}, []
    for weight_ce in (0, 1):
        def build(name, model, p):
            criterion, optimizer = ORIGINAL_LOSS_OPTIMIZER(name, model, p)
            criterion.loss.weight_ce = weight_ce
            return criterion, optimizer
        profiler.dice_only_optimizer = build
        arm_rows = []
        for geometry in ('ellipsoid', 'box'):
            row = profiler.profile('nnunet', 'dice', geometry, False, architecture, protocol)
            row['weight_ce'] = weight_ce
            rows.append(row)
            arm_rows.append(row)
            print(json.dumps({k: row[k] for k in ('weight_ce', 'geometry', 'flops')}), flush=True)
        measured = {r['flops'] for r in arm_rows}
        if len(measured) != 1:
            raise ValueError('Geometry changed the update cost')
        costs[str(weight_ce)] = measured.pop()
    write(args.output, dict(status='profiled', costs_by_ce_weight=costs, profiles=rows,
        parameters=rows[0]['parameters'], input_shape=[1, 1, 64, 64, 64],
        recipe='Native foreground Dice, identical deep supervision and SGD; CE weight alone toggled 0/1',
        runtime=dict(torch=audit.torch.__version__, monai=audit.monai.__version__)))


if __name__ == '__main__':
    main()
