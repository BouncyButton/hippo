"""Replay actual calibration and check stronger losses without training a model."""
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
STAGE = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else ROOT / "experiments/geofwi3d_constraint_strength_20261020"
model = sys.argv[1]
bundle = STAGE / "bundles" / ("unetr" if model == "unetr_vit_b16" else "basic")
sys.path[:0] = [str(bundle), str(bundle / "source"), str(bundle / "source/scripts"),
                str(bundle / "source/baselines/nnUNet")]
import worker
import torch
import numpy as np
from scripts.cross_dataset_campaign import conventional as exp
from scripts.adni_constraints import objectives as binary

torch.set_num_threads(2)
read = lambda p: json.loads(p.read_text())
plan = read(STAGE / "plan.json")
controls = read(STAGE / "control_calibrations.json")
records = []
for task in plan["tasks"]:
    if task["model"] != model or task.get('role') == 'confirmation':
        continue
    p = read(STAGE / "protocols" / task["id"] / "protocol.json")
    cal = controls[task["control_id"]]
    assert len(cal["cases"]) == 19
    original, _ = exp.select_at_target(cal["cases"], task["arm"], cal["target_ratio"], cal["p95_cap"])
    assert all(math.isclose(original[k], cal["weights"][k], rel_tol=1e-10) for k in original)
    selected, stats = exp.select_at_target(
        cal["cases"], task["arm"], p["calibration_target_ratio"], p["calibration_p95_cap"])
    assert all(math.isfinite(w) and w > original[k] for k, w in selected.items())
    records.append(dict(task=task["id"], original=original, selected=selected,
                        coefficient_multipliers={k: selected[k] / original[k] for k in selected},
                        statistics=stats))

# Actual foreground loss and backward at each distinct stronger setting.
loss_records = []
for task in [t for t in plan["tasks"] if t["model"] == model and t["seed"] == 0 and t.get('role') != 'confirmation']:
    p = read(STAGE / "protocols" / task["id"] / "protocol.json")
    out = STAGE / "checks" / task["id"]
    out.mkdir(parents=True, exist_ok=True)
    worker.configure(p, out)
    if model == "unetr_vit_b16":
        import constraint_objective as objective
        loss_fn = objective.training_loss
    else:
        loss_fn = binary.training_loss
    torch.manual_seed(18)
    coords = torch.meshgrid(*(torch.arange(96) for _ in range(3)), indexing="ij")
    y = ((((coords[0]-40)/24)**2 + ((coords[1]-43)/21)**2 + ((coords[2]-51)/35)**2) < 1)[None, None].long()
    z = torch.randn(1, 2, 96, 96, 96, requires_grad=True)
    criterion = None
    outputs = z
    if model == "nnunet":
        from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
        from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
        from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
        weights = np.array([1., .5, .25, 0.])
        criterion = DeepSupervisionWrapper(DC_and_CE_loss(
            {"batch_dice": True, "smooth": 1e-5, "do_bg": False, "ddp": False}, {},
            weight_ce=1, weight_dice=1, ignore_label=None, dice_class=MemoryEfficientSoftDiceLoss),
            weights / weights.sum())
        outputs = [z] + [torch.nn.functional.interpolate(
            z, size=(n, n, n), mode="trilinear", align_corners=False) for n in (48, 24, 12)]
    arm = task["arm"]
    sd = binary.signed_distance_maps(y) if "sdf" in arm else None
    depth = binary.geometry_weights(y, np.eye(4), p["depth_geometry"]) if "depth" in arm else None
    weights = next(r["selected"] for r in records if r["task"] == task["id"])
    loss = loss_fn(outputs, y, arm, weights, 1., sd, criterion, depth, 1.,
                   policy=p["sdf_policy"][arm])
    grad = torch.autograd.grad(loss, z)[0]
    assert torch.isfinite(loss) and torch.isfinite(grad).all() and grad.square().sum() > 0
    loss_records.append(dict(task=task["id"], loss=float(loss), gradient_rms=float(grad.square().mean().sqrt())))

worker.io.write(STAGE / f"strength_checks_{model}.json", dict(
    status="passed", optimizer_updates=0, model=model, calibration_replays=records, loss_checks=loss_records))
print(json.dumps(dict(status="passed", model=model, calibration_replays=len(records),
                      losses=len(loss_records), optimizer_updates=0)))
