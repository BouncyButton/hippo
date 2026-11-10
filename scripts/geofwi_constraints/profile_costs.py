"""CPU-only differential profiling of new UNETR auxiliary combinations."""
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/geofwi3d_constraints_3fold_3seed_20261020"
BUNDLE = ROOT / "reports/geofwi3d_frozen_3fold_3seed_20261020/frozen/bundles/unetr"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/hippo-mpl")
sys.path[:0] = [str(BUNDLE), str(BUNDLE / "source"), str(BUNDLE / "source/scripts"),
                str(BUNDLE / "source/baselines/nnUNet"), str(ROOT / "scripts")]
import rescue_objective as rescue
from scripts.cross_dataset_campaign.common import ARMS, digest
from scripts.adni_constraints.audit_correction import correct_row
from profile_intermediate_baselines import audit, intermediate_supplement
spec = importlib.util.spec_from_file_location("constraint_objective", Path(__file__).with_name("objective.py"))
objective = importlib.util.module_from_spec(spec)
spec.loader.exec_module(objective)
torch = audit.torch


def measure(labels, arm, calibration):
    z = torch.randn(1, 2, 96, 96, 96, requires_grad=True)
    sdf = objective.binary.signed_distance_maps(labels) if "sdf" in arm else None
    depth = torch.full((96, 96, 96), 1 / 96**3) if "depth" in arm else None
    operator = audit.reference.OperatorAudit()
    operator.stage = "calibration_logit_gradients" if calibration else "loss_logit_backward"
    with operator, audit.FlopCounterMode(display=False) as counter:
        if calibration:
            objective.calibration_case(z, labels, arm, sdf, None, depth)
        else:
            objective.training_loss(z, labels, arm, dict(bands=.04, edge=.1, sdf=.1, depth=.1),
                                    1., sdf, None, depth, .2).backward()
    return correct_row(dict(flops=counter.get_total_flops() + sum(operator.supplement.values()),
                            supplementary_per_stage=operator.supplement, operator_audit=operator.ops))


def main():
    assert torch.__version__.startswith("2.2.2") and audit.monai.__version__ == "1.6.0"
    torch.set_num_threads(2)
    torch.manual_seed(0)
    audit.reference.supplementary_flops = intermediate_supplement
    parent = json.loads((BUNDLE / "rescue_profile.json").read_text())
    full_baseline = parent["costs"]["dice"]["update_flops"]
    forwards = {r["core_flops_per_stage"]["forward"] + r["supplementary_per_stage"].get("forward", 0)
                for r in parent["profiles"] if r["calibration"]}
    assert len(forwards) == 1
    eval_forward = forwards.pop()
    rows, costs = [], {}
    arms = tuple(dict.fromkeys((*ARMS, "dice_edge", "dice_edge_sdf")))
    for arm in arms:
        for geometry in ("ellipsoid", "box"):
            labels = torch.nn.functional.interpolate((audit.synthetic_target(geometry) > 0).float(),
                         size=(96, 96, 96), mode="nearest").long().transpose(2, 3).contiguous()
            for cal in ([False] if arm == "dice" else [False, True]):
                row = dict(arm=arm, geometry=geometry, calibration=cal, **measure(labels, arm, cal))
                rows.append(row)
        train = {r["flops"] for r in rows if r["arm"] == arm and not r["calibration"]}
        cal = {r["flops"] for r in rows if r["arm"] == arm and r["calibration"]}
        assert len(train) == 1 and (not cal or len(cal) == 1)
        base = next(r["flops"] for r in rows if r["arm"] == "dice")
        costs[arm] = dict(update_flops=full_baseline + train.pop() - base,
                         calibration_flops=19 * (eval_forward + cal.pop()) +
                         (4096 if "depth" in arm else 1024) if cal else 0)
        print(json.dumps(dict(arm=arm, **costs[arm])), flush=True)
        (OUT / "unetr_profile.partial.json").write_text(json.dumps(dict(costs=costs, rows=rows), indent=2))
    assert costs["dice"] == parent["costs"]["dice"]
    assert costs["dice_bands"] == parent["costs"]["dice_bands"], (costs["dice_bands"], parent["costs"]["dice_bands"])
    result = dict(status="passed", costs=costs, profiles=rows, parameters=parent["parameters"],
                  parent_profile_sha256=digest(BUNDLE / "rescue_profile.json"),
                  objective_sha256=digest(Path(__file__).with_name("objective.py")),
                  method="Frozen full model baseline plus measured loss/logit-backward delta; "
                         "frozen eval forward plus actual calibration gradients over all 19 training cases. "
                         "Matched baseline and BANDS costs must reproduce the prior full graph profiles exactly.",
                  runtime=dict(torch=torch.__version__, monai=audit.monai.__version__))
    (OUT / "unetr_profile.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
