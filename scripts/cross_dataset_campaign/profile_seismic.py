"""Measure actual binary loss/update/calibration graphs with the MSD FLOP audit."""
import argparse
import gc
import json
from pathlib import Path
import numpy as np

from scripts.adni_baselines.core import ROOT, MODELS, base, build_model
from scripts.adni_constraints import objectives as obj
from scripts.cross_dataset_campaign import conventional
from scripts.capped_depth_delay import objectives as combined
def build_model(identity, architecture):
    return conventional.build_model(dict(model_identity=identity, dataset="GeoFWI3D"), architecture)
def depth_loss(logits, labels, weights):
    return (audit.torch.nn.functional.cross_entropy(logits.float(), labels[:, 0].long(), reduction="none")[0]*weights).sum()
combined.depth_loss = depth_loss
from scripts.adni_constraints.audit_correction import correct_row
from profile_intermediate_baselines import audit, intermediate_supplement


def measure(identity, architecture, protocol, arm, geometry, calibration):
    base.build_loss_and_optimizer = conventional.dice_only_optimizer
    torch = audit.torch
    torch.manual_seed(0)
    model = build_model(identity, architecture)
    criterion, optimizer = base.build_loss_and_optimizer(MODELS[identity]['alias'], model, protocol)
    native = criterion if identity == 'nnunet' else None
    if native is not None:
        for p in model.parameters():
            optimizer.state[p]['momentum_buffer'] = torch.zeros_like(p)
    model.eval() if calibration else model.train()
    image = torch.randn(1, 1, 96, 96, 96)
    # Rotate synthetic long axis off axis 0 so projected-axis depth is defined.
    target = torch.nn.functional.interpolate((audit.synthetic_target(geometry) > 0).float(), size=(96,96,96), mode='nearest').long().transpose(2,3).contiguous()
    sdf = obj.signed_distance_maps(target) if 'sdf' in arm else None
    depth = obj.geometry_weights(target, np.eye(4), obj.GEOMETRY) if 'depth' in arm else None
    operator = audit.reference.OperatorAudit()
    stages = {}
    with operator, audit.FlopCounterMode(display=False, custom_mapping={
            torch.ops.aten.convolution_backward: audit.reference.conv_backward_flops}) as counter:
        before = counter.get_total_flops()
        if calibration:
            with torch.no_grad():
                outputs = model(image)
        else:
            outputs = model(image)
        stages['forward'] = counter.get_total_flops() - before
        operator.stage = 'calibration_logit_gradients' if calibration else 'loss'
        before = counter.get_total_flops()
        if calibration:
            obj.calibration_case(outputs, target, arm, sdf, native, depth)
        else:
            loss = obj.training_loss(outputs, target, arm, dict(bands=.04, edge=.1, sdf=.1, depth=.1),
                                     1., sdf, native, depth, .2)
        stages[operator.stage] = counter.get_total_flops() - before
        if not calibration:
            operator.stage = 'backward'; before = counter.get_total_flops()
            loss.backward()
            stages['backward'] = counter.get_total_flops() - before
            operator.stage = 'optimizer'; before = counter.get_total_flops()
            if native is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 12, foreach=False, error_if_nonfinite=True)
            optimizer.step()
            stages['optimizer'] = counter.get_total_flops() - before
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite profile')
    row = dict(model=identity, arm=arm, geometry=geometry, calibration=calibration,
               flops=sum(stages.values()) + sum(operator.supplement.values()),
               parameters=sum(p.numel() for p in model.parameters()),
               core_flops_per_stage=stages, supplementary_per_stage=operator.supplement,
               operator_audit=operator.ops)
    del model, outputs, optimizer
    gc.collect()
    return correct_row(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--all-arms', action='store_true')
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if audit.monai.__version__ != '1.6.0' or not audit.torch.__version__.startswith('2.2.2'):
        raise ValueError('Use original local profiling runtime')
    audit.torch.set_num_threads(2)
    audit.reference.supplementary_flops = intermediate_supplement
    directory = ROOT/'experiments/adni_baselines_5pct_fold0_seed0_20261013/protocols'/args.model
    architecture = json.loads((directory/'nnunet_architecture.json').read_text())
    protocol = json.loads((directory/'protocol.json').read_text())
    rows, costs = [], {}
    for arm in ((*obj.ARMS, 'dice_edge', 'dice_edge_sdf') if args.all_arms else ('dice',)):
        measured = []
        for calibration in ([False] if arm == 'dice' else [False, True]):
            for geometry in ('ellipsoid', 'box'):
                row = measure(args.model, architecture, protocol, arm, geometry, calibration)
                rows.append(row); measured.append(row)
                print(json.dumps({k: row[k] for k in ('model','arm','geometry','calibration','flops')}), flush=True)
        train = {r['flops'] for r in measured if not r['calibration']}
        cal = {r['flops'] for r in measured if r['calibration']}
        if len(train) != 1 or (arm != 'dice' and len(cal) != 1):
            raise ValueError('Dense arithmetic depends on target geometry')
        costs[arm] = dict(update_flops=train.pop(),
                         calibration_flops=19 * cal.pop() + (4096 if 'depth' in arm else 1024) if cal else 0)
        base.atomic_json(args.output.with_suffix('.partial.json'), dict(costs=costs, rows=rows))
    base.atomic_json(args.output, dict(model=args.model, loss_recipe='dice_only', num_classes=2, input_shape=[1,1,96,96,96], costs=costs, profiles=rows,
        parameters=rows[0]['parameters'], runtime=dict(torch=audit.torch.__version__, monai=audit.monai.__version__),
        accounting='Original MSD dense arithmetic (MAC=2), actual model/loss/backward/optimizer graphs; calibration measured separately. Fixed annotation targets, validation, augmentation and I/O excluded. Host ramp arithmetic added by scheduler.'))


if __name__ == '__main__':
    main()
