"""Fold-0 experiment: freeze the canonical segmenter, learn spline corrections.

Run with ``python -m semantic_constraints.spline_refinement.train --help``.
No annotation-derived feature or edit support is passed to the regressor.
"""
import argparse
import csv
import hashlib
import inspect
import json
import random
import time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
import torch
from torch.utils.data import DataLoader
from monai.networks.nets import SwinUNETR

from baselines.swin_unetr.swin_unetr import _build_monai_dataset_from_pkl, _load_pkl_dataframe
from thesis.new_constraints.bands.outer_boundary import build_boundary_bands
from .model import CoefficientHead, SplineWarp, jacobian_determinant, refinement_loss

EXPECTED_CHECKPOINT = "abbc13be98a720d9c25ed2ee2aefb0a3a7f20811148734e7561886c63a93baf1"
EXPECTED_SPLITS = "1d6a3fe993359ad85e9c28a29901162251c3359449b710c09cb9e92d66dda472"
EXPECTED_DATA = "3311d2233b99e09080c40ca80f1035781d9c1dc9c5b109dc266c73eb52a3df01"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False))


def signed_distance(mask):
    return np.where(mask, -(ndi.distance_transform_edt(mask) - .5),
                    ndi.distance_transform_edt(~mask) - .5).astype(np.float32)


def make_features(image, probability, prediction):
    """All arguments come from the image or frozen prediction, never the label."""
    mask = prediction > 0
    inner, outer = build_boundary_bands(torch.from_numpy(mask)[None, None], steps=2)
    inner, outer = inner[0, 0].numpy(), outer[0, 0].numpy()
    support = inner | outer
    distance = signed_distance(mask)
    t = np.clip(ndi.distance_transform_edt(support) / 2, 0, 1)
    return {
        "inputs": torch.from_numpy(np.stack((image, probability, np.clip(distance / 4, -1, 1), inner, outer)).astype(np.float32)),
        "source_sdf": torch.from_numpy(distance[None]),
        "support": torch.from_numpy(support[None].astype(np.float32)),
        "taper": torch.from_numpy((t * t * (3 - 2 * t))[None].astype(np.float32)),
        "prediction": torch.from_numpy(prediction.astype(np.uint8)),
    }


def prepare(args, device):
    fold = json.loads(args.splits.read_text())[0]
    assert len(fold["train"]) == 208 and len(fold["val"]) == 52
    assert not set(fold["train"]) & set(fold["val"])
    hashes = {"checkpoint": sha256(args.checkpoint), "splits": sha256(args.splits), "pkl": sha256(args.pkl)}
    assert hashes["checkpoint"] == EXPECTED_CHECKPOINT, hashes
    assert hashes["splits"] == EXPECTED_SPLITS, hashes
    assert hashes["pkl"] == EXPECTED_DATA, hashes
    dataset = _build_monai_dataset_from_pkl(_load_pkl_dataframe(args.pkl), "MSD", 3)
    options = dict(in_channels=1, out_channels=3, use_checkpoint=False, drop_rate=0.)
    if "img_size" in inspect.signature(SwinUNETR).parameters:
        options["img_size"] = (64, 64, 64)
    baseline = SwinUNETR(**options).to(device).eval()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    baseline.load_state_dict(payload.get("model", payload), strict=True)
    baseline.requires_grad_(False)
    del payload
    records = {"train": [], "val": []}
    raw_shapes = {}
    start = time.time()
    with torch.no_grad():
        for index in range(len(dataset)):
            name = dataset.data[index]["case_name"]
            cohort = "train" if name in fold["train"] else "val"
            assert name in fold[cohort]
            if args.smoke and len(records[cohort]) >= 2:
                continue
            sample = dataset[index]
            image = sample["image"].as_tensor().float()
            labels = sample["label"].as_tensor()[0].numpy().astype(np.uint8)
            assert image.shape == (1, 64, 64, 64)
            # FP32 once for both baseline and refinement comparisons.
            logits = baseline(image[None].to(device)).float()
            prediction = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
            probability = logits.softmax(1)[0, 1:].sum(0).cpu().numpy()
            record = make_features(image[0].numpy(), probability, prediction)
            record.update(name=name, labels=torch.from_numpy(labels),
                          truth=torch.from_numpy((labels > 0)[None].astype(np.float32)),
                          target_sdf=torch.from_numpy(signed_distance(labels > 0)[None]))
            records[cohort].append(record)
            raw_shapes[name] = list(dataset.data[index]["image"].shape)
            if (index + 1) % 20 == 0:
                print(f"CACHE {index + 1}/{len(dataset)} elapsed={time.time() - start:.1f}s", flush=True)
    del baseline, dataset
    if device.type == "cuda":
        torch.cuda.empty_cache()
    assert all({r["name"] for r in records[c]} == set(fold[c]) for c in records) or args.smoke
    return records, dict(hashes=hashes, split=fold, raw_shapes=raw_shapes,
                         cache_seconds=time.time() - start)


def move(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def dice(prediction, truth):
    return float((2 * np.count_nonzero(prediction & truth) + 1e-8) /
                 (np.count_nonzero(prediction) + np.count_nonzero(truth) + 1e-8))


def restore_labels(mask, original):
    labels = np.where(mask, original, 0).astype(np.uint8)
    added = mask & (original == 0)
    if added.any():
        assert (original > 0).any()
        indices = ndi.distance_transform_edt(original == 0, return_distances=False, return_indices=True)
        nearest = original[tuple(indices)]
        labels[added] = nearest[added]
    return labels


def surface_distances(prediction, truth):
    a = prediction & ~ndi.binary_erosion(prediction)
    b = truth & ~ndi.binary_erosion(truth)
    if not a.any() or not b.any():
        return None, None
    distances = np.r_[ndi.distance_transform_edt(~b)[a], ndi.distance_transform_edt(~a)[b]]
    return float(distances.mean()), float(np.percentile(distances, 95))


@torch.no_grad()
def evaluate(model, geometry, loader, device, output=None):
    model.eval()
    rows = []
    for batch in loader:
        batch = move(batch, device)
        coefficients = model(batch["inputs"])
        warped, backward, velocity = geometry(coefficients, batch["source_sdf"], batch["taper"], batch["support"])
        determinants = jacobian_determinant(backward)
        for index, name in enumerate(batch["name"]):
            original = batch["prediction"][index].cpu().numpy()
            baseline = original > 0
            truth = batch["truth"][index, 0].cpu().numpy().astype(bool)
            support = batch["support"][index, 0].cpu().numpy().astype(bool)
            raw_mask = warped[index, 0].cpu().numpy() < 0
            det = determinants[index]
            valid = bool(torch.isfinite(det).all() and torch.isfinite(warped[index]).all() and det.min() > .05)
            # Predeclared, annotation-independent safeguard, never a GT-based choice.
            mask = raw_mask if valid else baseline.copy()
            assert not np.any((mask != baseline) & ~support)
            row = dict(case=name, baseline_dice=dice(baseline, truth), raw_refined_dice=dice(raw_mask, truth),
                       refined_dice=dice(mask, truth), valid_geometry=valid,
                       jacobian_min=float(det.min()) if torch.isfinite(det).all() else None,
                       max_displacement_mm=float(backward[index].square().sum(0).sqrt().max()),
                       baseline_fp=int((baseline & ~truth).sum()), baseline_fn=int((~baseline & truth).sum()),
                       refined_fp=int((mask & ~truth).sum()), refined_fn=int((~mask & truth).sum()),
                       fixed=int(((baseline != truth) & (mask == truth)).sum()),
                       introduced=int(((baseline == truth) & (mask != truth)).sum()),
                       changed_outside_band=int(((mask != baseline) & ~support).sum()),
                       errors_in_band=int(((baseline != truth) & support).sum()))
            row["delta_dice"] = row["refined_dice"] - row["baseline_dice"]
            if output is not None:
                labels = restore_labels(mask, original)
                gt_labels = batch["labels"][index].cpu().numpy()
                retained = mask & baseline
                row["retained_ap_changes"] = int((labels[retained] != original[retained]).sum())
                row["baseline_ap_mean_dice"] = np.mean([dice(original == k, gt_labels == k) for k in (1, 2)]).item()
                row["refined_ap_mean_dice"] = np.mean([dice(labels == k, gt_labels == k) for k in (1, 2)]).item()
                for key, value in (("baseline", baseline), ("refined", mask)):
                    assd, hd95 = surface_distances(value, truth)
                    row[f"{key}_voxel_surface_assd_mm"] = assd
                    row[f"{key}_voxel_surface_hd95_mm"] = hd95
                np.savez_compressed(output / f"{name}.npz", prediction=original, refined=labels,
                                    truth=gt_labels, image=batch["inputs"][index, 0].cpu().numpy(),
                                    coefficients=coefficients[index].cpu().numpy(), support=support)
            rows.append(row)
    return rows


def summarize(rows):
    delta = np.asarray([r["delta_dice"] for r in rows])
    samples = np.random.default_rng(0).choice(delta, (5000, len(delta)), replace=True).mean(1)
    result = dict(cases=len(rows), improved=int((delta > 1e-10).sum()), worsened=int((delta < -1e-10).sum()),
                  unchanged=int((np.abs(delta) <= 1e-10).sum()), fallback_cases=sum(not r["valid_geometry"] for r in rows),
                  mean_delta_dice=float(delta.mean()), median_delta_dice=float(np.median(delta)),
                  descriptive_paired_bootstrap_95ci=np.quantile(samples, [.025, .975]).tolist())
    for key in ("baseline_dice", "refined_dice", "raw_refined_dice", "baseline_ap_mean_dice", "refined_ap_mean_dice",
                "baseline_voxel_surface_assd_mm", "refined_voxel_surface_assd_mm",
                "baseline_voxel_surface_hd95_mm", "refined_voxel_surface_hd95_mm"):
        if key in rows[0]:
            values = [r[key] for r in rows if r[key] is not None]
            result[f"mean_{key}"] = float(np.mean(values)) if values else None
    for key in ("baseline_fp", "baseline_fn", "refined_fp", "refined_fn", "fixed", "introduced", "changed_outside_band", "errors_in_band", "retained_ap_changes"):
        if key in rows[0]:
            result[key] = sum(r[key] for r in rows)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pkl", "splits", "checkpoint", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    records, provenance = prepare(args, device)
    geometry = SplineWarp().to(device)
    model = CoefficientHead(geometry.control_shape).to(device)
    train_loader = DataLoader(records["train"], batch_size=args.batch_size, shuffle=True, num_workers=0)
    train_eval = DataLoader(records["train"], batch_size=args.batch_size, num_workers=0)
    val_loader = DataLoader(records["val"], batch_size=args.batch_size, num_workers=0)
    protocol = dict(**provenance, seed=0, augmentation="none", train_cases=len(records["train"]),
        val_cases=len(records["val"]), parameters=sum(p.numel() for p in model.parameters()),
        control_shape=geometry.control_shape, control_spacing_mm=4, squarings=6,
        inputs=["MRI", "foreground_probability", "clipped_source_sdf_divided_by_4", "predicted_inner_band", "predicted_outer_band"],
        bands="2 steps, 6 connectivity, initial prediction only; smoothstep EDT/2 taper; hard preserve outside",
        optimizer="AdamW lr .001 weight_decay .0001, gradient norm clip 5, FP32",
        loss="band smooth-L1 SDF + .25 soft Dice + .005 velocity L2 + .02 velocity gradient L2 + mean relu(.3-Jacobian)^2 in band",
        selection="highest mean validation union Dice including identity epoch 0; patience 8 for gain .0005 after minimum 25, maximum 50 epochs",
        safeguard="per-case identity fallback if sampled backward Jacobian <= .05 or nonfinite; uses no annotation",
        labels="retain AP on retained initial foreground; nearest initial label for additions",
        limitations=["in-sample frozen baseline predictions on training cohort", "validation used for early stopping and prior development", "sampled positive Jacobians do not prove continuous or voxel topology preservation", "single seed"],
        torch_version=torch.__version__, device=str(device),
        gpu=torch.cuda.get_device_name() if device.type == "cuda" else None,
        source_hashes={str(p.name): sha256(p) for p in Path(__file__).parent.glob("*.py")})
    write_json(args.output / "protocol.json", protocol)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    history = []
    initial_rows = evaluate(model, geometry, val_loader, device)
    assert all(abs(r["delta_dice"]) < 1e-10 and r["valid_geometry"] for r in initial_rows)
    best_score = patience_score = summarize(initial_rows)["mean_refined_dice"]
    best_epoch = stale = 0
    torch.save(dict(model=model.state_dict(), epoch=0), args.output / "best.pt")
    history.append(dict(epoch=0, validation=summarize(initial_rows)))
    start = time.time()
    for epoch in range(1, (2 if args.smoke else args.epochs) + 1):
        model.train()
        totals, loss_sum, count = {}, 0., 0
        for batch in train_loader:
            batch = move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            coefficients = model(batch["inputs"])
            warped, backward, velocity = geometry(coefficients, batch["source_sdf"], batch["taper"], batch["support"])
            determinant = jacobian_determinant(backward)
            loss, parts = refinement_loss(warped, batch["target_sdf"], batch["truth"], batch["support"], velocity, determinant)
            assert torch.isfinite(loss), "Nonfinite training objective"
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            size = len(batch["name"])
            loss_sum += float(loss.detach()) * size
            count += size
            for key, value in parts.items():
                totals[key] = totals.get(key, 0.) + value * size
        validation = summarize(evaluate(model, geometry, val_loader, device))
        score = validation["mean_refined_dice"]
        if score > best_score:
            best_score, best_epoch = score, epoch
            torch.save(dict(model=model.state_dict(), epoch=epoch), args.output / "best.pt")
        if score > patience_score + .0005:
            patience_score, stale = score, 0
        else:
            stale += 1
        row = dict(epoch=epoch, train_loss=loss_sum / count, train_parts={k: v / count for k, v in totals.items()},
                   validation=validation, best_epoch=best_epoch, elapsed_seconds=time.time() - start)
        history.append(row)
        write_json(args.output / "history.json", history)
        print(json.dumps(row), flush=True)
        if epoch >= 25 and stale >= 8:
            break
    torch.save(dict(model=model.state_dict(), epoch=epoch), args.output / "last.pt")
    last_validation = summarize(evaluate(model, geometry, val_loader, device))
    model.load_state_dict(torch.load(args.output / "best.pt", map_location=device, weights_only=False)["model"])
    cases_dir = args.output / "validation_cases"
    cases_dir.mkdir()
    rows = evaluate(model, geometry, val_loader, device, cases_dir)
    write_json(args.output / "validation_cases.json", rows)
    with (args.output / "validation_cases.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    results = dict(best_epoch=best_epoch, stopped_epoch=epoch, training_seconds=time.time() - start,
                   validation=summarize(rows), last_validation=last_validation,
                   train_at_selected_epoch=summarize(evaluate(model, geometry, train_eval, device)),
                   case_164=next((r for r in rows if r["case"].endswith("164")), None))
    write_json(args.output / "results.json", results)
    print("COMPLETE " + json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
