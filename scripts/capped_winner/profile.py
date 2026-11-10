"""Conservative sparse-loss arithmetic bound under the existing FLOP convention."""
import argparse
import json
from pathlib import Path
import torch
from scripts.profile_intermediate_baselines import audit, intermediate_supplement
from .objectives import auxiliary, calibration_case, ARM
from .geometry import worst_case_targets


def measure(function):
    operator_audit = audit.reference.OperatorAudit()
    operator_audit.stage = 'extra_loss_and_logit_backward'
    with operator_audit, audit.FlopCounterMode(display=False) as counter:
        function()
    return dict(flops=counter.get_total_flops()+sum(operator_audit.supplement.values()),
                operator_audit=operator_audit.ops)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    audit.reference.supplementary_flops = intermediate_supplement
    z = torch.randn(1, 3, 64, 64, 64, requires_grad=True)
    labels = audit.synthetic_target('ellipsoid')
    target = worst_case_targets()
    def backward():
        terms = auxiliary(z, labels, target, upper_bound=True)
        loss = sum(.1 * term for term in terms.values())
        loss.backward()
    train = measure(backward)
    calibration = measure(lambda: calibration_case(z, labels, ARM, depth_weights=target, upper_bound=True))
    # Joining the extra and native gradients costs at most one addition per
    # full-resolution logit. Host coefficient/ramp arithmetic is bounded too.
    bound = train['flops'] + z.numel() + 128
    report = dict(extra_update_flops_bound=bound,
        extra_calibration_flops_bound=10*calibration['flops']+65536,
        train=train, calibration=calibration,
        runtime=dict(torch=torch.__version__, monai=audit.monai.__version__),
        method='Reuse frozen per-model native-update counts; add a separately audited upper bound for the new loss and its logit backward. N voxels per band side; 6N directed endpoints per side per scale. Count the extra gradient addition. All actual target cardinalities are bounded by construction.',
        exclusions='Same prior convention: validation, target preprocessing, augmentation, I/O, integer operations, comparisons, indexing and top-k ranking excluded. MAC=2.',
        caution='Charged FLOPs are conservative arithmetic estimates, not GPU instruction measurements. Calibration is reported separately and outside the optimization cap.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('train','calibration')}), flush=True)


if __name__ == '__main__':
    main()
