"""Profile the actual extra CE/calibration graphs using the established audit."""
import argparse
import gc
import json
from pathlib import Path
import numpy as np

from campaign90 import MODELS, ROOT, build_model, digest, write_json
from profile_intermediate_baselines import audit, intermediate_supplement
from scripts.capped_depth_delay.objectives import DEPTH_ARMS, training_loss, calibration_case, geometry_weights
from thesis.new_constraints.signed_distance import signed_distance_maps
import train_compute_matched_baselines as base
from scripts.cross_dataset_campaign.conventional import dice_only_optimizer


def profile(identity, arm, geometry, calibration, architecture, protocol):
    base.build_loss_and_optimizer = dice_only_optimizer
    torch = audit.torch
    torch.manual_seed(0)
    model = build_model(identity, architecture)
    alias = MODELS[identity]['alias']
    criterion, optimizer = base.build_loss_and_optimizer(alias, model, protocol)
    native = criterion if alias == 'nnunet' else None
    if native is not None:
        for parameter in model.parameters():
            optimizer.state[parameter]['momentum_buffer'] = torch.zeros_like(parameter)
    model.eval() if calibration else model.train()
    image = torch.randn(1, 1, 64, 64, 64)
    target = audit.synthetic_target(geometry)
    sdf = signed_distance_maps(target) if 'sdf' in arm else None
    # The audit ellipsoid/box have PC1 along array axis 0. Map that axis to
    # anatomical anterior, rather than the right axis excluded by the original
    # plane definition. The integer target/loss graphs stay unchanged.
    affine = np.eye(4)
    affine[:3, :3] = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    depth = geometry_weights(target, affine, protocol['depth_geometry']) if 'depth' in arm else None
    weights = dict(bands=.04, edge=.1, sdf=.1, depth=.1)
    operator_audit = audit.reference.OperatorAudit()
    stages = {}
    with operator_audit, audit.FlopCounterMode(display=False, custom_mapping={
            torch.ops.aten.convolution_backward: audit.reference.conv_backward_flops}) as counter:
        before = counter.get_total_flops()
        if calibration:
            with torch.no_grad():
                outputs = model(image)
        else:
            outputs = model(image)
        stages['forward'] = counter.get_total_flops() - before
        operator_audit.stage = 'calibration_logit_gradients' if calibration else 'loss'
        before = counter.get_total_flops()
        if calibration:
            calibration_case(outputs, target, arm, sdf, native, depth)
        else:
            loss = training_loss(outputs, target, arm, weights, 1., sdf, native, depth, .2)
        stages[operator_audit.stage] = counter.get_total_flops() - before
        if not calibration:
            operator_audit.stage = 'backward'
            before = counter.get_total_flops()
            loss.backward()
            stages['backward'] = counter.get_total_flops() - before
            operator_audit.stage = 'optimizer'
            before = counter.get_total_flops()
            if native is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 12, foreach=False, error_if_nonfinite=True)
            optimizer.step()
            stages['optimizer'] = counter.get_total_flops() - before
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite profile loss')
    row = dict(model=identity, arm=arm, geometry=geometry, calibration=calibration,
               flops=sum(stages.values()) + sum(operator_audit.supplement.values()),
               parameters=sum(p.numel() for p in model.parameters()),
               core_flops_per_stage=stages, supplementary_per_stage=operator_audit.supplement,
               operator_audit=operator_audit.ops)
    del model, optimizer, outputs
    gc.collect()
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if audit.monai.__version__ != '1.6.0' or not audit.torch.__version__.startswith('2.2.2'):
        raise ValueError('Use the established local profiling runtime')
    audit.torch.set_num_threads(2)
    audit.reference.supplementary_flops = intermediate_supplement
    origin = ROOT / 'experiments/nnunet_pair95_20261005/baseline'
    architecture = json.loads((origin / 'nnunet_architecture.json').read_text())
    protocol = json.loads((origin / 'protocol.json').read_text())
    protocol['initial_lrs'].update({s['alias']: s['initial_lr'] for s in MODELS.values()})
    protocol['depth_geometry'] = json.loads((ROOT / 'docs/experiments/depth_swinunetr_20261006/protocol.json').read_text())['geometry']
    from scripts.cross_dataset_campaign.common import ARMS
    costs, rows = {}, []
    for arm in (*ARMS, 'dice_edge', 'dice_edge_sdf'):
        measured = []
        for calibration in ([False] if arm == 'dice' else [False, True]):
            for geometry in ('ellipsoid', 'box'):
                row = profile(args.model, arm, geometry, calibration, architecture, protocol)
                rows.append(row); measured.append(row)
                print(json.dumps({k: row[k] for k in ('model','arm','geometry','calibration','flops')}), flush=True)
        train = {r['flops'] for r in measured if not r['calibration']}
        cal = {r['flops'] for r in measured if r['calibration']}
        if len(train) != 1 or arm != 'dice' and len(cal) != 1:
            raise ValueError('Geometry changed arithmetic')
        costs[arm] = dict(update_flops=train.pop(), calibration_flops=10*cal.pop()+(4096 if 'depth' in arm else 1024) if cal else 0)
        write_json(args.output.with_suffix('.partial.json'), dict(costs=costs, profiles=rows))

    write_json(args.output, dict(status='profiled', model=args.model, costs=costs, profiles=rows,
        accounting='Established dense arithmetic MAC=2; forward/loss/backward/optimizer charged. Calibration measured separately. Original full-volume SDF unchanged. Fixed annotation geometry, augmentation, validation and I/O excluded as target preprocessing. Independent ramp host operations added by scheduling.',
        runtime=dict(torch=audit.torch.__version__, monai=audit.monai.__version__),
        input_shape=[1, 1, 64, 64, 64]))


if __name__ == '__main__':
    main()
