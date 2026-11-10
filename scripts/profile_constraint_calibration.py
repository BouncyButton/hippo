#!/usr/bin/env python3
"""Charge ten training-only forward/logit-gradient calibration cases per arm."""
import argparse
import hashlib
import json
from pathlib import Path

import profile_unet_constraint_budget as audit
from compute_constraint_objectives import calibration_case
from thesis.new_constraints.signed_distance import signed_distance_maps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("unet", "swinunetr"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert audit.monai.__version__ == "1.6.0" and audit.torch.__version__.startswith("2.2.2")
    audit.torch.set_num_threads(4)
    audit.reference.supplementary_flops = audit.constraint_supplement
    profiles = []
    for arm in audit.ARMS[1:]:
        for geometry in ("ellipsoid", "box"):
            audit.torch.manual_seed(0)
            model, _ = audit.matched.build_network(args.model, "matched")
            model.eval()
            image = audit.torch.randn(1, 1, 64, 64, 64)
            target = audit.synthetic_target(geometry)
            sdf = signed_distance_maps(target) if "sdf" in arm else None
            operator_audit = audit.reference.OperatorAudit()
            with operator_audit, audit.FlopCounterMode(display=False, custom_mapping={
                audit.torch.ops.aten.convolution_backward: audit.reference.conv_backward_flops,
            }) as counter:
                with audit.torch.no_grad():
                    raw = model(image)
                operator_audit.stage = "calibration_logit_gradients"
                calibration_case(raw, target, arm, sdf)
            cost = counter.get_total_flops() + sum(operator_audit.supplement.values())
            profiles.append({"arm": arm, "geometry": geometry, "flops_per_case": cost,
                             "core_flops": counter.get_total_flops(),
                             "supplementary_estimate_per_stage": operator_audit.supplement,
                             "operator_audit": operator_audit.ops})
            print(json.dumps({"model": args.model, "arm": arm, "geometry": geometry, "flops_per_case": cost}), flush=True)
    costs = {"dice": 0}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / (args.model + "_calibration_audit.json")).write_text(json.dumps(profiles, indent=2) + "\n")
    for arm in audit.ARMS[1:]:
        per_case = {p["flops_per_case"] for p in profiles if p["arm"] == arm}
        assert len(per_case) == 1, "Mask geometry changed calibration arithmetic"
        # Explicit conservative allowance for scalar coefficient selection,
        # quantile interpolation and the combined-gradient norm identity.
        costs[arm] = 10 * max(per_case) + 1024
    paths = [Path(__file__), Path(calibration_case.__code__.co_filename),
             audit.ROOT / "thesis/new_constraints/signed_distance.py"]
    result = {"model": args.model, "costs": costs, "profiles": profiles,
              "source_sha256": {str(p.relative_to(audit.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
              "calibration_patients": 10, "postprocessing_scalar_flop_allowance": 1024,
              "scope": "One eval-mode no-grad model forward per training patient; FP32 gradients of each loss with respect to detached logits and RMS/cross-moment statistics; no model backward or optimizer update"}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / (args.model + "_calibration_profiles.json")).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(costs, indent=2), flush=True)


if __name__ == "__main__":
    main()
