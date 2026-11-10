"""Profile the exact baseline, bands update, and training-only calibration graphs."""
import argparse
import gc
import json
from pathlib import Path

import profile_unet_constraint_budget as audit
import train_compute_matched_baselines as base
from baselines.residual_unet import build_unet48
from compute_constraint_objectives import training_loss, calibration_case
import nnunet_band_objective as native


def profile(model_name, protocol_dir, arm, geometry, calibration=False):
    protocol = json.loads((protocol_dir / 'protocol.json').read_text())
    architecture = json.loads((protocol_dir / 'nnunet_architecture.json').read_text())
    audit.torch.manual_seed(0)
    model = build_unet48() if model_name == 'unet' else base.build_model(model_name, architecture)
    criterion, optimizer = base.build_loss_and_optimizer(model_name, model, protocol)
    if model_name == 'nnunet':
        for parameter in model.parameters():
            optimizer.state[parameter]['momentum_buffer'] = audit.torch.zeros_like(parameter)
    model.eval() if calibration else model.train()
    image = audit.torch.randn(1, 1, 64, 64, 64)
    target = audit.synthetic_target(geometry)
    operator_audit = audit.reference.OperatorAudit()
    stages = {}
    with operator_audit, audit.FlopCounterMode(display=False, custom_mapping={
            audit.torch.ops.aten.convolution_backward: audit.reference.conv_backward_flops}) as counter:
        before = counter.get_total_flops()
        if calibration:
            with audit.torch.no_grad():
                outputs = model(image)
        else:
            outputs = model(image)
        stages['forward'] = counter.get_total_flops() - before
        operator_audit.stage = 'calibration_logit_gradients' if calibration else 'loss'
        before = counter.get_total_flops()
        if calibration:
            if model_name == 'nnunet':
                native.calibration_case(outputs, target, arm, criterion)
            else:
                calibration_case(outputs, target, arm)
        else:
            if model_name == 'nnunet':
                loss = native.training_loss(outputs, target, arm, {'bands': .04}, .2, criterion)
            else:
                loss = training_loss(outputs, target, arm, {'bands': .04}, .2)
        stages[operator_audit.stage] = counter.get_total_flops() - before
        if not calibration:
            operator_audit.stage = 'backward'
            before = counter.get_total_flops()
            loss.backward()
            stages['backward'] = counter.get_total_flops() - before
            operator_audit.stage = 'optimizer'
            before = counter.get_total_flops()
            if model_name == 'nnunet':
                audit.torch.nn.utils.clip_grad_norm_(model.parameters(), 12, foreach=False, error_if_nonfinite=True)
            optimizer.step()
            stages['optimizer'] = counter.get_total_flops() - before
            if not audit.torch.isfinite(loss):
                raise ValueError('Nonfinite profile loss')
    cost = sum(stages.values()) + sum(operator_audit.supplement.values())
    result = {'model': model_name, 'arm': arm, 'geometry': geometry,
              'calibration': calibration, 'flops': cost,
              'parameters': sum(p.numel() for p in model.parameters()),
              'core_flops_per_stage': stages,
              'supplementary_estimate_per_stage': operator_audit.supplement,
              'operator_audit': operator_audit.ops}
    del model, optimizer, criterion, outputs
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['unet', 'swinunetr', 'nnunet'], required=True)
    parser.add_argument('--baseline-protocol-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if audit.monai.__version__ != '1.6.0' or not audit.torch.__version__.startswith('2.2.2'):
        raise ValueError('Use the established profiling runtime')
    audit.torch.set_num_threads(4)
    audit.reference.supplementary_flops = audit.constraint_supplement
    rows = []
    for arm, calibration in [('dice', False), ('dice_bands', False), ('dice_bands', True)]:
        for geometry in ['ellipsoid', 'box']:
            row = profile(args.model, args.baseline_protocol_dir, arm, geometry, calibration)
            rows.append(row)
            print(json.dumps({k: row[k] for k in ['model', 'arm', 'geometry', 'calibration', 'flops']}), flush=True)
    costs = {}
    for key, arm, calibration in [('baseline', 'dice', False), ('bands', 'dice_bands', False),
                                  ('calibration_per_case', 'dice_bands', True)]:
        distinct = {r['flops'] for r in rows if r['arm'] == arm and r['calibration'] == calibration}
        if len(distinct) != 1:
            raise ValueError('Geometry changed the declared arithmetic count')
        costs[key] = distinct.pop()
    original = json.loads((args.baseline_protocol_dir / 'protocol.json').read_text())['models'][args.model]
    expected = original['objectives']['dice']['flops_per_update'] if args.model == 'unet' else original['flops_per_update']
    if costs['baseline'] != expected:
        raise ValueError(f'Profile changed the baseline cost: {costs["baseline"]} versus {expected}')
    costs['calibration_total'] = 10 * costs['calibration_per_case'] + 1024
    sources = [Path(__file__), Path(native.__file__), audit.ROOT / 'scripts/train_compute_matched_constraints.py',
               audit.ROOT / 'scripts/train_compute_matched_baselines.py',
               audit.ROOT / 'scripts/compute_constraint_objectives.py', Path(audit.__file__),
               audit.PROFILE_ROOT / 'profile_reference.py', audit.PROFILE_ROOT / 'profile_matched_models.py',
               audit.ROOT / 'baselines/residual_unet.py',
               audit.ROOT / 'thesis/new_constraints/bands/outer_boundary.py',
               audit.ROOT / 'thesis/new_constraints/constraint_result.py']
    result = {'model': args.model, 'costs': costs, 'profiles': rows,
              'runtime': {'torch': audit.torch.__version__, 'monai': audit.monai.__version__},
              'input_shape': [1, 1, 64, 64, 64], 'parameters': rows[0]['parameters'],
              'source_sha256': {str(p.relative_to(audit.ROOT)): base.digest(p) for p in sources},
              'calibration_training_volumes': 10, 'scalar_selection_allowance_flops': 1024,
              'accounting': 'Established dense arithmetic convention, MAC=2; validation, data preparation, integer/boolean/comparison and memory operations excluded. SGD recurring momentum cost bounds first update.',
              'nnunet_calibration_scope': 'Full-resolution native Dice+CE gradient includes its normalized deep-supervision weight; bands applied only to this output. Other native heads retained.'}
    args.output.mkdir(parents=True, exist_ok=True)
    base.atomic_json(args.output / (args.model + '_profiles.json'), result)
    print(json.dumps(costs), flush=True)


if __name__ == '__main__':
    main()
