"""Training-case frozen-logit audit of cumulative ray supervision.

No network optimization and no validation-label access. Per-case normalized
updates compare directions; they are not calibrated training coefficients.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
import torch
import torch.nn.functional as F

from thesis.new_constraints.ray_moments import RayMomentLoss
from .losses import RayGeometry, grouped_masks, wrong_foreground_bce

METHODS = ("dice", "ce", "wrong_fg_bce", "legacy_m0", "legacy_m01",
           "axis_thickness", "axis_depth", "multi_thickness", "multi_depth",
           "multi_thickness_depth")
KINDS = ("correct", "translation", "missing_tip", "protrusion", "hole", "ap_shift")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def dice_loss(logits, labels):
    p = logits.softmax(1)
    g = F.one_hot(labels, 3).movedim(-1, 1).to(p)
    return (1 - (2 * (p*g).sum((2, 3, 4)) + 1e-5) /
            (p.sum((2, 3, 4)) + g.sum((2, 3, 4)) + 1e-5)).mean()


def loss_for(name, logits, labels, geometry):
    if name == "dice":
        return dice_loss(logits, labels)
    if name == "ce":
        return F.cross_entropy(logits, labels)
    if name == "wrong_fg_bce":
        return wrong_foreground_bce(logits, labels)
    if name.startswith("legacy"):
        return RayMomentLoss(orders=(0,) if name == "legacy_m0" else (0, 1))(logits, labels).loss
    pred, truth = grouped_masks(logits.softmax(1), labels)
    if name == "multi_thickness_depth":
        return geometry.loss(pred, truth, "thickness") + geometry.loss(pred, truth, "depth")
    views, mode = name.split("_")
    return geometry.loss(pred, truth, mode, include_oblique=views == "multi")


def synthetic_logits(candidate, *, confidence_scale=1.):
    """Exact candidate argmax, lower margin at its interfaces, finite interiors."""
    distance = np.zeros_like(candidate, dtype=np.float32)
    for class_id in range(3):
        mask = candidate == class_id
        distance[mask] = ndi.distance_transform_edt(mask)[mask]
    margin = np.minimum(20., .5 + 1.5 * distance) * confidence_scale
    logits = np.eye(3, dtype=np.float32)[candidate] * margin[..., None]
    return torch.from_numpy(logits).movedim(-1, 0)[None]


def corruptions(truth):
    """Fixed perturbations of actual training masks, never selected by outcomes."""
    yield "correct", truth.copy()
    shifted = np.zeros_like(truth)
    shifted[2:] = truth[:-2]
    yield "translation", shifted
    coords = np.argwhere(truth > 0)
    if not len(coords):
        raise ValueError("Empty training annotation")
    tip = truth.copy()
    tip[:, :int(coords[:, 1].min()) + 3] = 0
    yield "missing_tip", tip
    foreground = truth > 0
    grid = np.indices(truth.shape)
    center = coords[np.argmax(coords[:, 0])].copy()
    center[0] += 1
    ball = ((grid - center[:, None, None, None])**2).sum(0) <= 3**2
    protrusion = truth.copy()
    nearest = ndi.distance_transform_edt(~foreground, return_distances=False, return_indices=True)
    add = ball & ~foreground
    protrusion[add] = truth[tuple(nearest)][add]
    yield "protrusion", protrusion
    center = np.array(np.unravel_index(ndi.distance_transform_edt(foreground).argmax(), truth.shape))
    hole = truth.copy()
    hole[((grid - center[:, None, None, None])**2).sum(0) <= 2**2] = 0
    yield "hole", hole
    ap = truth.copy()
    ap[ndi.binary_dilation(truth == 1, iterations=2) & (truth == 2)] = 1
    yield "ap_shift", ap


def scores(logits, labels):
    pred = logits.argmax(1)
    foreground, truth_fg = pred != 0, labels != 0
    def dice(a, b):
        return float((2 * (a & b).sum() + 1e-8) / (a.sum() + b.sum() + 1e-8))
    class_dice = [dice(pred == c, labels == c) for c in (1, 2)]
    return dict(fg_dice=dice(foreground, truth_fg), ap_dice=sum(class_dice)/2,
                anterior_dice=class_dice[0], posterior_dice=class_dice[1],
                fp=int((foreground & ~truth_fg).sum()), fn=int((~foreground & truth_fg).sum()),
                swaps=int((foreground & truth_fg & (pred != labels)).sum()),
                errors=int((pred != labels).sum()), soft_dice_loss=float(dice_loss(logits, labels)),
                ce=float(F.cross_entropy(logits, labels)))


def error_masks(logits, labels):
    pred = logits.argmax(1)
    correct = pred == labels
    return dict(correct_background=correct & (labels == 0),
                correct_foreground=correct & (labels != 0),
                fp=(pred != 0) & (labels == 0), fn=(pred == 0) & (labels != 0),
                swaps=(pred != 0) & (labels != 0) & ~correct)


def rms(value):
    return value.double().square().mean().sqrt()


def gradient_details(logits, labels, grad, dice_grad, ce_grad):
    if not torch.isfinite(grad).all():
        raise RuntimeError("Nonfinite gradient")
    pred = logits.argmax(1)
    descent = -grad
    true_change = descent.gather(1, labels[:, None])[:, 0]
    other_logits = logits.detach().clone()
    other_logits.scatter_(1, labels[:, None], -torch.inf)
    competitor = other_logits.argmax(1)
    margin_change = true_change - descent.gather(1, competitor[:, None])[:, 0]
    energy = grad.double().square().sum(1)
    total = float(energy.sum())
    masks = error_masks(logits, labels)
    masks["confident_wrong"] = (pred != labels) & (logits.softmax(1).amax(1) >= .99)
    def cosine(other):
        denom = float(grad.double().norm() * other.double().norm())
        return float((grad.double()*other.double()).sum()) / denom if denom else None
    strata = {}
    for kind, mask in masks.items():
        count = int(mask.sum())
        strata[kind] = dict(count=count, helpful=int(((margin_change > 0) & mask).sum()),
                            harmful=int(((margin_change < 0) & mask).sum()),
                            zero=int(((margin_change == 0) & mask).sum()),
                            energy=float(energy[mask].sum()) / total if total else 0.)
    return dict(rms=float(rms(grad)), max_abs=float(grad.abs().max()),
                cosine_dice=cosine(dice_grad), cosine_ce=cosine(ce_grad), strata=strata,
                raw_energy=total,
                channel_energy=[float(grad[:, c].double().square().sum()) / total if total else 0.
                                for c in range(3)])


def normalized(grad, norm, size):
    denominator = rms(grad) if norm == "rms" else grad.abs().max()
    return grad * (size / denominator) if denominator > 0 else torch.zeros_like(grad)


def step_record(logits, labels, grad, norm, size, baseline):
    update = -normalized(grad, norm, size)
    after_logits = logits + update
    before, after = logits.argmax(1), after_logits.argmax(1)
    was_correct, is_correct = before == labels, after == labels
    was_fg_correct = (before != 0) == (labels != 0)
    is_fg_correct = (after != 0) == (labels != 0)
    metrics = scores(after_logits, labels)
    fixed = int((~was_correct & is_correct).sum())
    introduced = int((was_correct & ~is_correct).sum())
    assert metrics["errors"] == baseline["errors"] - fixed + introduced
    details = dict(normalization=norm, step=size, fixed=fixed, introduced=introduced,
                   external_fixed=int((~was_fg_correct & is_fg_correct).sum()),
                   external_introduced=int((was_fg_correct & ~is_fg_correct).sum()),
                   update_rms=float(rms(update)), update_max=float(update.abs().max()),
                   after=metrics, delta={key: metrics[key] - baseline[key] for key in metrics})
    details["fixed_by_true_class"] = [int((~was_correct & is_correct & (labels == c)).sum()) for c in range(3)]
    details["introduced_by_true_class"] = [int((was_correct & ~is_correct & (labels == c)).sum()) for c in range(3)]
    return details


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        for step in row["steps"]:
            key = (row["kind"], row["method"], step["normalization"], step["step"])
            grouped[key].append((row, step))
    results = []
    for (kind, method, norm, size), items in sorted(grouped.items()):
        result = dict(kind=kind, method=method, normalization=norm, step=size, cases=len(items))
        for name in ("fixed", "introduced", "external_fixed", "external_introduced"):
            result[name] = sum(s[name] for _, s in items)
        for name in ("fp", "fn", "swaps", "errors", "fg_dice", "ap_dice", "anterior_dice",
                     "posterior_dice", "soft_dice_loss", "ce"):
            result["delta_"+name] = float(np.mean([s["delta"][name] for _, s in items]))
        changes = [s["delta"]["errors"] for _, s in items]
        result.update(fewer_errors=sum(v < 0 for v in changes), more_errors=sum(v > 0 for v in changes),
                      unchanged=sum(v == 0 for v in changes))
        cosines = [r["gradient"]["cosine_dice"] for r, _ in items
                   if r["gradient"]["cosine_dice"] is not None]
        result["median_cosine_dice"] = float(np.median(cosines)) if cosines else None
        result["median_gradient_rms"] = float(np.median([r["gradient"]["rms"] for r, _ in items]))
        result["gradient_strata"] = {}
        for stratum in items[0][0]["gradient"]["strata"]:
            entries = [r["gradient"]["strata"][stratum] for r, _ in items]
            count = sum(v["count"] for v in entries)
            result["gradient_strata"][stratum] = dict(
                count=count, helpful_fraction=sum(v["helpful"] for v in entries)/count if count else None,
                harmful_fraction=sum(v["harmful"] for v in entries)/count if count else None,
                mean_energy_fraction=float(np.mean([v["energy"] for v in entries])))
        results.append(result)
    return results


def load_cases(args, device):
    # Load only training rows into the transform pipeline. Validation labels
    # necessarily coexist in the historical pickle but are not transformed,
    # inspected, used as loss targets, scored, or selected here.
    from baselines.swin_unetr.swin_unetr import _build_monai_dataset_from_pkl, _load_pkl_dataframe, _case_name_from_row
    from monai.networks.nets import SwinUNETR
    split = json.loads(args.splits.read_text())[0]
    if set(split["train"]) & set(split["val"]):
        raise ValueError("Overlapping train and validation case IDs")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_info = {k: checkpoint[k] for k in ("configuration", "config", "step", "epoch", "best_step", "best_score") if k in checkpoint}
    configuration = checkpoint.get("configuration", checkpoint.get("config", {}))
    fitted = sorted(configuration.get("train_cases", []))
    if not set(fitted).issubset(split["train"]):
        raise ValueError("Checkpoint training cases do not belong to requested training fold")
    eligible = sorted(set(split["train"]) - set(fitted))
    rng = np.random.default_rng(0)
    if len(fitted) <= args.cases:
        names = sorted(fitted + rng.choice(eligible, size=min(args.cases-len(fitted), len(eligible)), replace=False).tolist())
    else:
        names = sorted(rng.choice(fitted, size=args.cases, replace=False).tolist())
    df = _load_pkl_dataframe(args.pkl)
    selected = df.apply(lambda row: _case_name_from_row(row, "MSD") in set(names), axis=1)
    dataset = _build_monai_dataset_from_pkl(df[selected].copy(), "MSD", 3)
    del df
    options = dict(in_channels=1, out_channels=3, use_checkpoint=False, drop_rate=0.)
    if "img_size" in inspect.signature(SwinUNETR).parameters:
        options["img_size"] = (64, 64, 64)
    model = SwinUNETR(**options).to(device).eval().requires_grad_(False)
    model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
    del checkpoint
    records = []
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            name = dataset.data[i]["case_name"]
            image = sample["image"].as_tensor().float()[None].to(device)
            labels = sample["label"].as_tensor().long()[0]
            if tuple(labels.shape) != (64, 64, 64):
                raise ValueError(labels.shape)
            logits = model(image).float().cpu()
            records.append(dict(name=name, labels=labels.cpu(), logits=logits,
                                checkpoint_training_case=name in fitted))
            print("INFERENCE", name, i+1, "/", len(dataset), flush=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if set(names) != {r["name"] for r in records}:
        raise ValueError("Missing selected cases")
    return sorted(records, key=lambda r: r["name"]), dict(selected_names=names,
                selection="Include checkpoint training cases up to requested count; fill remaining slots from other fold-0 training IDs with NumPy seed 0, before outputs",
                selected_checkpoint_training_names=sorted(set(names) & set(fitted)),
                fold=0, full_train_count=len(split["train"]), val_count=len(split["val"]),
                validation_used=False, checkpoint_metadata=checkpoint_info)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("pkl", "splits", "checkpoint", "output"):
        parser.add_argument("--"+key, type=Path, required=True)
    parser.add_argument("--cases", type=int, default=32)
    parser.add_argument("--synthetic-cases", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.cases < 1 or args.synthetic_cases < 0:
        raise ValueError("Invalid cohort sizes")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "inputs").mkdir()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    start = time.time()
    sources = [Path(__file__), Path(__file__).with_name("losses.py"), Path(__file__).with_name("test_losses.py"),
               Path("thesis/new_constraints/ray_moments.py"), Path("thesis/new_constraints/constraint_result.py"),
               Path("baselines/swin_unetr/swin_unetr.py")]
    hashes = {str(p): sha256(p) for p in (args.pkl, args.splits, args.checkpoint, *sources)}
    records, provenance = load_cases(args, device)
    for record in records:
        np.savez_compressed(args.output / "inputs" / (record["name"] + ".npz"),
                            logits=record["logits"].numpy(), labels=record["labels"].numpy())
    protocol = dict(**provenance, hashes=hashes, torch_version=torch.__version__, device=str(device),
        methods=list(METHODS), source_loss="Dice including background, smoothing 1e-5",
        class_groups=[[1, 2], [1], [2]], group_weighting="equal",
        ray_geometry="3 exact axes; 4 body diagonals with detector spacing 2 mm, depth spacing 1 mm, full support, fixed crop coordinates",
        physical_spacing_mm=[1, 1, 1], length_normalization_mm=64,
        depth_loss="L1 mean of all inclusive prefix and suffix residual integrals; full profiles, no depth subsampling",
        combined_loss="thickness + depth with coefficients 1 and 1 before diagnostic normalization",
        steps={"max": [.25, 1., 2.], "rms": [.02, .05, .10]},
        combined_update="Dice + auxiliary rescaled to 10% of same-case Dice gradient RMS, then total RMS .10",
        synthetic_names=[r["name"] for r in records[:args.synthetic_cases]],
        synthetic_kinds=list(KINDS), additional_saturation_check="first 2 synthetic cases, translated logits at 4x margin",
        interpretation="GT-supervised local logit diagnostics only; not inference repair, SGD, or proof of training benefit; no validation labels used",
        historical_controls="Unmodified legacy RayMomentLoss M0 and M0+M1; plain wrong-only foreground BCE as in September 15 reevaluation",
        exclusions="No gradients through MRI, network weights, sampling geometry or target; no per-case outcome selection")
    write_json(args.output / "protocol.json", protocol)
    geometry = RayGeometry((64, 64, 64), device=device)
    rows = []
    for index, record in enumerate(records):
        y = record["labels"][None].to(device)
        scenarios = [("real", record["logits"])]
        if index < args.synthetic_cases:
            scenarios += [(kind, synthetic_logits(candidate)) for kind, candidate in corruptions(record["labels"].numpy())]
        if index < min(2, args.synthetic_cases):
            shifted = dict(corruptions(record["labels"].numpy()))["translation"]
            scenarios.append(("translation_saturated", synthetic_logits(shifted, confidence_scale=4)))
        for kind, source in scenarios:
            logits = source.to(device).requires_grad_()
            baseline = scores(logits.detach(), y)
            dice_grad, = torch.autograd.grad(dice_loss(logits, y), logits)
            ce_grad, = torch.autograd.grad(F.cross_entropy(logits, y), logits)
            for method in METHODS:
                tick = time.time()
                loss = loss_for(method, logits, y, geometry)
                grad, = torch.autograd.grad(loss, logits)
                details = gradient_details(logits.detach(), y, grad, dice_grad, ce_grad)
                steps = []
                for norm, sizes in (("max", (.25, 1., 2.)), ("rms", (.02, .05, .10))):
                    for size in sizes:
                        steps.append(step_record(logits.detach(), y, grad, norm, size, baseline))
                if method != "dice":
                    aux = grad * (0.1 * rms(dice_grad) / rms(grad)) if rms(grad) > 0 else grad
                    combined = step_record(logits.detach(), y, dice_grad + aux, "rms", .1, baseline)
                    combined["normalization"] = "dice_plus_10pct_rms"
                    steps.append(combined)
                after_small = logits.detach() - normalized(grad, "max", 1e-3)
                with torch.no_grad():
                    own_after = float(loss_for(method, after_small, y, geometry))
                row = dict(case=record["name"], kind=kind, method=method, baseline=baseline,
                           checkpoint_training_case=record["checkpoint_training_case"],
                           loss=float(loss.detach()), loss_after_max_0p001=own_after,
                           gradient=details, steps=steps, seconds=time.time()-tick)
                rows.append(row)
            print("PROBE", index+1, record["name"], kind, "errors", baseline["errors"],
                  "elapsed", round(time.time()-start, 1), flush=True)
        write_json(args.output / "rows.json", rows)
    summary = summarize(rows)
    write_json(args.output / "summary.json", summary)
    write_json(args.output / "completion.json", dict(rows=len(rows), cases=len(records),
               elapsed_seconds=time.time()-start, output_sha256={f: sha256(args.output/f)
                   for f in ("protocol.json", "rows.json", "summary.json")}))
    for row in summary:
        if row["kind"] == "real" and row["normalization"] in ("rms", "dice_plus_10pct_rms") and row["step"] == .1:
            print("SUMMARY", json.dumps({k:v for k,v in row.items() if k != "gradient_strata"}), flush=True)


if __name__ == "__main__":
    main()
