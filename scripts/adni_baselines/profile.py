"""Profile the binary models with the original MSD arithmetic convention."""
import argparse
import json
from pathlib import Path
import sys

from .core import BUDGET, MODELS, ROOT, base, build_model

sys.path.insert(0, str(ROOT / 'reports/compute_budget_5pct_fold0_seed0_20261003'))
import profile_matched_models as matched
import profile_reference as reference
import monai
import torch
from torch.utils.flop_counter import FlopCounterMode


def profile(identity, architecture, classes):
    torch.manual_seed(0)
    model = build_model(identity, architecture)
    if classes == 3:
        if identity in ('swinunetr', 'nnunet'):
            old_arch = json.loads(json.dumps(architecture))
            old_arch['kwargs']['num_classes'] = 3
            model = base.build_model(MODELS[identity]['alias'], old_arch)
        else:
            from baselines.simple_segmentation import build_candidate
            model = build_candidate(identity)
    model.train()
    p = dict(initial_lrs={MODELS[identity]['alias']:MODELS[identity]['lr']},
             adamw_weight_decay=1e-5, sgd_weight_decay=3e-5, nnunet_sgd_momentum=.95)
    criterion, optimizer = base.build_loss_and_optimizer(MODELS[identity]['alias'], model, p)
    if identity == 'nnunet':
        for parameter in model.parameters():
            optimizer.state[parameter]['momentum_buffer'] = torch.zeros_like(parameter)
    image = torch.randn(1, 1, 64, 64, 64)
    target = torch.zeros(1, 1, 64, 64, 64, dtype=torch.long)
    target[:, :, 20:40, 20:40, 20:40] = 1
    if classes == 3:
        target[:, :, 30:40, 20:40, 20:40] = 2
    audit, stages = reference.OperatorAudit(), {}
    with audit, FlopCounterMode(display=False, custom_mapping={
        torch.ops.aten.convolution_backward: reference.conv_backward_flops}) as counter:
        before = counter.get_total_flops()
        output = model(image)
        stages['forward'] = counter.get_total_flops() - before
        audit.stage = 'loss'; before = counter.get_total_flops()
        if identity == 'nnunet':
            targets = [torch.nn.functional.interpolate(target.float(), size=o.shape[2:], mode='nearest').long() for o in output]
            loss = criterion(output, targets)
        else:
            loss = criterion(output, target) + .2 * (output.float().reshape(-1)[0] * 0.)
        stages['loss'] = counter.get_total_flops() - before
        audit.stage = 'backward'; before = counter.get_total_flops()
        loss.backward()
        stages['backward'] = counter.get_total_flops() - before
        audit.stage = 'optimizer'; before = counter.get_total_flops()
        if identity == 'nnunet':
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12, foreach=False)
        optimizer.step()
        stages['optimizer'] = counter.get_total_flops() - before
    if not torch.isfinite(loss) or not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('Nonfinite real forward/backward/optimizer smoke check')
    cost = sum(stages.values()) + sum(audit.supplement.values())
    return dict(model=identity, classes=classes, parameters=sum(p.numel() for p in model.parameters()),
                flops_per_update=cost, max_steps=BUDGET // cost,
                core_flops_per_stage=stages, supplementary_flops=audit.supplement,
                operator_audit=audit.ops, loss=float(loss.detach()),
                runtime=dict(torch=torch.__version__, monai=monai.__version__),
                next_update_exceeds_budget=(BUDGET // cost + 1) * cost > BUDGET)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=list(MODELS), required=True)
    parser.add_argument('--classes', type=int, choices=[2, 3], default=2)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if monai.__version__ != '1.6.0' or not torch.__version__.startswith('2.2.2'):
        raise RuntimeError('Use original MSD profiling runtime torch 2.2.2 / MONAI 1.6.0')
    torch.set_num_threads(4)
    architecture = json.loads((ROOT / 'experiments/compute_matched_baselines_20261003/nnunet_architecture.json').read_text())
    architecture['kwargs']['num_classes'] = 2
    result = profile(args.model, architecture, args.classes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    base.atomic_json(args.output, result)
    print(json.dumps({k:result[k] for k in ('model','classes','parameters','flops_per_update','max_steps','loss')}), flush=True)


if __name__ == '__main__':
    main()
