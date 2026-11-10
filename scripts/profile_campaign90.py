"""Measure every production objective and calibration graph on two geometries."""
import argparse
import gc
import json

from campaign90 import ARMS, MODELS, ROOT, build_model, digest, write_json
from profile_intermediate_baselines import audit, intermediate_supplement
from compute_constraint_objectives import training_loss, calibration_case
from thesis.new_constraints.signed_distance import signed_distance_maps
import train_compute_matched_baselines as base
import nnunet_multi_objective as native


def profile(identity, arm, geometry, calibration, architecture, protocol):
    torch = audit.torch
    torch.manual_seed(0)
    model = build_model(identity, architecture)
    alias = MODELS[identity]['alias']
    criterion, optimizer = base.build_loss_and_optimizer(alias, model, protocol)
    if alias == 'nnunet':
        for p in model.parameters():
            optimizer.state[p]['momentum_buffer'] = torch.zeros_like(p)
    model.eval() if calibration else model.train()
    image, target = torch.randn(1, 1, 64, 64, 64), audit.synthetic_target(geometry)
    sdf = signed_distance_maps(target) if 'sdf' in arm else None
    weights = {'bands': .04, 'edge': .1, 'sdf': .1}
    operator_audit = audit.reference.OperatorAudit()
    stages = {}
    with operator_audit, audit.FlopCounterMode(display=False, custom_mapping={
        torch.ops.aten.convolution_backward: audit.reference.conv_backward_flops,
    }) as counter:
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
            if alias == 'nnunet':
                native.calibration_case(outputs, target, arm, criterion, sdf)
            else:
                calibration_case(outputs, target, arm, sdf)
        else:
            loss = (native.training_loss(outputs, target, arm, weights, .2, criterion, sdf)
                    if alias == 'nnunet' else training_loss(outputs, target, arm, weights, .2, sdf))
        stages[operator_audit.stage] = counter.get_total_flops() - before
        if not calibration:
            operator_audit.stage = 'backward'
            before = counter.get_total_flops()
            loss.backward()
            stages['backward'] = counter.get_total_flops() - before
            operator_audit.stage = 'optimizer'
            before = counter.get_total_flops()
            if alias == 'nnunet':
                torch.nn.utils.clip_grad_norm_(model.parameters(), 12, foreach=False, error_if_nonfinite=True)
            optimizer.step()
            stages['optimizer'] = counter.get_total_flops() - before
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite profile loss')
    result = {'model': identity, 'arm': arm, 'geometry': geometry, 'calibration': calibration,
              'flops': sum(stages.values()) + sum(operator_audit.supplement.values()),
              'parameters': sum(p.numel() for p in model.parameters()),
              'core_flops_per_stage': stages, 'supplementary_per_stage': operator_audit.supplement,
              'operator_audit': operator_audit.ops}
    del model, optimizer, outputs
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if audit.monai.__version__ != '1.6.0' or not audit.torch.__version__.startswith('2.2.2'):
        raise ValueError('Use the established profiling runtime')
    audit.torch.set_num_threads(2)
    audit.reference.supplementary_flops = intermediate_supplement
    origin = ROOT / 'experiments/nnunet_pair95_20261005/baseline'
    architecture = json.loads((origin / 'nnunet_architecture.json').read_text())
    protocol = json.loads((origin / 'protocol.json').read_text())
    protocol['initial_lrs'].update({s['alias']: s['initial_lr'] for s in MODELS.values()})
    rows, costs = [], {}
    for arm in ARMS:
        arm_rows = []
        for calibration in ([False] if arm == 'dice' else [False, True]):
            for geometry in ['ellipsoid', 'box']:
                row = profile(args.model, arm, geometry, calibration, architecture, protocol)
                rows.append(row)
                arm_rows.append(row)
                print(json.dumps({k: row[k] for k in ['model','arm','geometry','calibration','flops']}), flush=True)
        train_costs = {r['flops'] for r in arm_rows if not r['calibration']}
        cal_costs = {r['flops'] for r in arm_rows if r['calibration']}
        if len(train_costs) != 1 or arm != 'dice' and len(cal_costs) != 1:
            raise ValueError('Geometry changed arithmetic count')
        costs[arm] = {'update_flops': train_costs.pop(),
                      'calibration_flops': 10 * cal_costs.pop() + 1024 if cal_costs else 0}
    if costs['dice']['update_flops'] != MODELS[args.model]['baseline_flops']:
        raise ValueError('Baseline graph changed: ' + repr(costs['dice']))
    if any(r['parameters'] != MODELS[args.model]['parameters'] for r in rows):
        raise ValueError('Architecture changed')
    names = ['scripts/profile_campaign90.py', 'scripts/campaign90.py',
             'scripts/nnunet_multi_objective.py', 'scripts/nnunet_band_objective.py',
             'scripts/train_compute_matched_baselines.py', 'scripts/compute_constraint_objectives.py',
             'scripts/profile_intermediate_baselines.py', 'scripts/profile_unet_constraint_budget.py',
             'baselines/simple_segmentation.py', 'thesis/new_constraints/bands/outer_boundary.py',
             'thesis/new_constraints/edge_consistency.py', 'thesis/new_constraints/signed_distance.py',
             'thesis/new_constraints/constraint_result.py',
             'reports/compute_budget_5pct_fold0_seed0_20261003/profile_reference.py',
             'reports/compute_budget_5pct_fold0_seed0_20261003/profile_matched_models.py']
    write_json(args.output, {'status': 'profiled', 'model': args.model, 'costs': costs,
               'parameters': MODELS[args.model]['parameters'], 'profiles': rows,
               'runtime': {'torch': audit.torch.__version__, 'monai': audit.monai.__version__},
               'input_shape': [1,1,64,64,64], 'source_sha256': {p: digest(ROOT / p) for p in names},
               'accounting': 'Established dense arithmetic, MAC=2. Forward/loss/backward/optimizer and calibration included; validation, augmentation, SDF target preprocessing, I/O, integer/boolean/comparison operations and precision housekeeping excluded.',
               'calibration_cases': 10, 'scalar_selection_allowance_flops': 1024})


if __name__ == '__main__':
    main()
