"""Fit eight or all training subjects to 32-mode anatomical oracle corrections.

The PCA and mean-error offset use all 208 training subjects, as in the preceding
analysis. Validation subjects are excluded from fitting.
"""
import argparse
import hashlib
import inspect
import json
import time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
import torch
from torch import nn
from torch.nn import functional as F
from monai.networks.nets import SwinUNETR

from baselines.swin_unetr.swin_unetr import _build_monai_dataset_from_pkl, _load_pkl_dataframe
from semantic_constraints.spline_refinement.train import (
    EXPECTED_CHECKPOINT, EXPECTED_DATA, EXPECTED_SPLITS, sha256, signed_distance,
    dice, surface_distances, write_json, restore_labels,
)
from semantic_constraints.spectral_shape.audit import SIZE, fit_pca, band_fit, shift_into_canvas


class CoefficientRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        blocks = []
        previous = 5
        for width in (16, 32, 64, 64):
            blocks += [nn.Conv3d(previous, width, 3, stride=2, padding=1),
                       nn.GroupNorm(4, width), nn.GELU()]
            previous = width
        self.encoder = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.AdaptiveAvgPool3d(4), nn.Flatten(),
                                  nn.Linear(64 * 4 ** 3, 128), nn.GELU(), nn.Linear(128, 32))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, inputs):
        return self.head(self.encoder(inputs))


def translate_image(array, shift):
    """Translate into the common canvas; permit cropping distant image context."""
    output = np.zeros((SIZE,) * 3, np.float32)
    source, target = [], []
    for size, delta in zip(array.shape, shift):
        lo, hi = max(0, -int(delta)), min(size, SIZE - int(delta))
        source.append(slice(lo, hi))
        target.append(slice(lo + int(delta), hi + int(delta)))
    output[tuple(target)] = array[tuple(source)]
    return output


def select_cases(records):
    ranked = sorted(records, key=lambda r: (int((r["prediction"] > 0).sum()), r["name"]))
    rng = np.random.default_rng(0)
    selected = []
    for quartile in range(4):
        indices = rng.choice(np.arange(quartile * 52, (quartile + 1) * 52), size=2, replace=False)
        selected.extend(ranked[int(i)]["name"] for i in indices)
    assert len(set(selected)) == 8
    return selected


def prepare_training(args, device):
    hashes = dict(checkpoint=sha256(args.checkpoint), pkl=sha256(args.pkl), splits=sha256(args.splits))
    assert hashes == dict(checkpoint=EXPECTED_CHECKPOINT, pkl=EXPECTED_DATA, splits=EXPECTED_SPLITS), hashes
    fold = json.loads(args.splits.read_text())[0]
    assert len(fold["train"]) == 208 and len(fold["val"]) == 52
    assert not set(fold["train"]) & set(fold["val"])
    dataset = _build_monai_dataset_from_pkl(_load_pkl_dataframe(args.pkl), "MSD", 3)
    kwargs = dict(in_channels=1, out_channels=3, use_checkpoint=False, drop_rate=0.)
    if "img_size" in inspect.signature(SwinUNETR).parameters:
        kwargs["img_size"] = (64, 64, 64)
    model = SwinUNETR(**kwargs).to(device).eval().requires_grad_(False)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state.get("model", state), strict=True)
    del state
    records = []
    with torch.no_grad():
        for index, item in enumerate(dataset.data):
            if item["case_name"] not in fold["train"]:
                continue
            sample = dataset[index]
            image = sample["image"].as_tensor().float()
            logits = model(image[None].to(device)).float()
            records.append(dict(name=item["case_name"], image=image[0].numpy(),
                                truth=sample["label"].as_tensor()[0].numpy().astype(np.uint8),
                                prediction=logits.argmax(1)[0].cpu().numpy().astype(np.uint8),
                                probability=logits.softmax(1)[0, 1:].sum(0).cpu().numpy()))
            if len(records) % 40 == 0:
                print("TRAIN_CACHE", len(records), "/208", flush=True)
    del model, dataset
    torch.cuda.empty_cache()
    assert {r["name"] for r in records} == set(fold["train"])
    return records, dict(hashes=hashes, split=fold)


def field_objective(coefficients, cases):
    """Compare induced boundary-band fields, not arbitrary coefficient units."""
    relative_errors, dice_losses = [], []
    for index, case in enumerate(cases):
        delta = case["design"] @ coefficients[index] + case["offset"]
        relative_errors.append((delta - case["oracle_delta"]).square().mean() / case["energy"])
        probability = torch.sigmoid(-(case["source_band"] + delta) / .5)
        intersection = (probability * case["truth_band"]).sum() + case["outside_intersection"]
        size = probability.sum() + case["outside_prediction"] + case["truth_size"]
        dice_losses.append(1 - (2 * intersection + 1) / (size + 1))
    field_error = torch.stack(relative_errors).mean()
    boundary_loss = torch.stack(dice_losses).mean()
    return field_error + .05 * boundary_loss, field_error, boundary_loss


@torch.no_grad()
def evaluate(model, inputs, cases, original_modes, common_scale, error_mean, output=None):
    model.eval()
    normalized = torch.cat([model(batch) for batch in inputs.split(8)])
    actual_coefficients = normalized / common_scale[None]
    fields = actual_coefficients @ original_modes + error_mean
    rows = []
    for index, case in enumerate(cases):
        delta = fields[index].cpu().numpy().reshape((SIZE,) * 3)
        prediction, truth, support = case["prediction"], case["truth"], case["support"]
        mask = np.where(support, case["source_sdf"] + delta < 0, prediction)
        assert np.array_equal(mask[~support], prediction[~support])
        learned_labels = restore_labels(mask, case["initial_labels"])
        assert np.array_equal(learned_labels[~support], case["initial_labels"][~support])
        retained = prediction & mask
        assert np.array_equal(learned_labels[retained], case["initial_labels"][retained])
        predicted_band = case["design"] @ normalized[index] + case["offset"]
        relative_mse = float((predicted_band - case["oracle_delta"]).square().mean() / case["energy"])
        row = dict(case=case["name"], baseline_dice=case["baseline_dice"], oracle_dice=case["oracle_dice"],
                   learned_dice=dice(mask, truth), relative_oracle_field_mse=relative_mse,
                   fixed=int(((prediction != truth) & (mask == truth)).sum()),
                   introduced=int(((prediction == truth) & (mask != truth)).sum()),
                   fp=int((mask & ~truth).sum()), fn=int((~mask & truth).sum()),
                   outside_band_changes=int(((mask != prediction) & ~support).sum()))
        if output is not None:
            for label, shape in (("baseline", prediction), ("oracle", case["oracle_mask"]), ("learned", mask)):
                assd, hd95 = surface_distances(shape, truth)
                row[f"{label}_assd_mm"], row[f"{label}_hd95_mm"] = assd, hd95
            row["learned_components"] = ndi.label(mask)[1]
            row["baseline_components"] = ndi.label(prediction)[1]
            row["oracle_components"] = ndi.label(case["oracle_mask"])[1]
            np.savez_compressed(output / f"{case['name']}.npz", prediction=case["initial_labels"],
                truth=case["truth_labels"], refined=learned_labels, oracle=case["oracle_mask"],
                support=support, image=inputs[index, 0].cpu().numpy(), shift=case["shift"],
                coefficients=actual_coefficients[index].cpu().numpy(),
                oracle_coefficients=case["oracle_coefficients"].cpu().numpy())
        rows.append(row)
    baseline = np.mean([r["baseline_dice"] for r in rows])
    oracle = np.mean([r["oracle_dice"] for r in rows])
    learned = np.mean([r["learned_dice"] for r in rows])
    summary = dict(baseline_dice=float(baseline), oracle_dice=float(oracle), learned_dice=float(learned),
                   oracle_gain=float(oracle-baseline), learned_gain=float(learned-baseline),
                   oracle_gain_recovered=float((learned-baseline)/(oracle-baseline)) if oracle > baseline+1e-8 else None,
                   relative_oracle_field_mse=float(np.mean([r["relative_oracle_field_mse"] for r in rows])),
                   fixed=sum(r["fixed"] for r in rows), introduced=sum(r["introduced"] for r in rows),
                   improved=sum(r["learned_dice"] > r["baseline_dice"] for r in rows),
                   worsened=sum(r["learned_dice"] < r["baseline_dice"] for r in rows))
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("checkpoint", "pkl", "splits", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--all-training", action="store_true")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")
    start = time.time()
    records, provenance = prepare_training(args, device)
    selected = [r["name"] for r in records] if args.all_training else select_cases(records)
    if args.all_training:
        assert set(selected) == set(provenance["split"]["train"])
        assert len(selected) == 208 and args.epochs > 0 and args.batch_size > 0
    batches_per_epoch = (len(selected) + args.batch_size - 1) // args.batch_size
    total_steps = args.epochs * batches_per_epoch if args.all_training else args.steps
    evaluation_interval = 5 * batches_per_epoch if args.all_training else 25
    assert not set(selected) & set(provenance["split"]["val"])
    print("SELECTED", selected, flush=True)
    cases, train_shapes, train_errors = [], [], []
    for record in records:
        p = record["prediction"] > 0
        shift = np.rint((SIZE-1)/2 - np.argwhere(p).mean(0)).astype(int)
        initial_labels = shift_into_canvas(record["prediction"], shift)
        truth_labels = shift_into_canvas(record["truth"], shift)
        p, g = initial_labels > 0, truth_labels > 0
        source_sdf = np.clip(signed_distance(p), -8, 8)
        target_sdf = np.clip(signed_distance(g), -8, 8)
        residual = target_sdf - source_sdf
        train_shapes.append(target_sdf.ravel())
        train_errors.append(residual.ravel())
        if record["name"] in selected:
            inner = p & ~ndi.binary_erosion(p, iterations=2)
            outer = ndi.binary_dilation(p, iterations=2) & ~p
            support = inner | outer
            features = np.stack((translate_image(record["image"], shift),
                translate_image(record["probability"], shift), np.clip(source_sdf/4, -1, 1), inner, outer)).astype(np.float32)
            cases.append(dict(name=record["name"], prediction=p, truth=g, initial_labels=initial_labels,
                truth_labels=truth_labels, source_sdf=source_sdf, support=support, residual=residual,
                features=features, shift=shift, baseline_dice=dice(p, g)))
    del records
    shapes = torch.from_numpy(np.stack(train_shapes)).to(device)
    _, modes, explained = fit_pca(shapes, rank=32)
    error_mean = torch.from_numpy(np.mean(np.stack(train_errors), axis=0)).to(device)
    del shapes, train_shapes, train_errors
    assert modes.shape == (32, SIZE**3)
    # Scale modes using only the selected predicted bands, never oracle targets.
    union = np.logical_or.reduce([c["support"] for c in cases]).ravel()
    common_scale = modes[:, torch.from_numpy(np.flatnonzero(union)).to(device)].square().mean(1).sqrt().clamp_min(1e-8)
    for case in cases:
        support, p, g = case["support"], case["prediction"], case["truth"]
        idx = torch.from_numpy(np.flatnonzero(support)).to(device)
        original_design = modes[:, idx].T
        target = torch.from_numpy(case["residual"][support]).to(device)
        offset = error_mean[idx]
        oracle_delta, oracle_coefficients = band_fit(original_design, target, offset)
        case.update(design=original_design/common_scale[None], offset=offset,
            oracle_delta=oracle_delta, oracle_coefficients=oracle_coefficients,
            energy=(oracle_delta-offset).square().mean().clamp_min(1e-4),
            source_band=torch.from_numpy(case["source_sdf"][support]).to(device),
            truth_band=torch.from_numpy(g[support].astype(np.float32)).to(device),
            outside_intersection=int((p & g & ~support).sum()),
            outside_prediction=int((p & ~support).sum()), truth_size=int(g.sum()))
        case["oracle_mask"] = p.copy()
        case["oracle_mask"][support] = case["source_sdf"][support] + oracle_delta.cpu().numpy() < 0
        case["oracle_dice"] = dice(case["oracle_mask"], g)
        # Equivalent coefficient scalings must induce the same physical field.
        torch.testing.assert_close(case["design"] @ (oracle_coefficients*common_scale)+offset,
                                   oracle_delta, atol=1e-5, rtol=1e-5)
    inputs = torch.from_numpy(np.stack([c.pop("features") for c in cases])).to(device)
    model = CoefficientRegressor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.0003)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=.000003)
    selection = "all 208 fold-0 training subjects" if args.all_training else "two cases per predicted-volume quartile; seed 0; no oracle-based selection"
    protocol = dict(**provenance, selected=[c["name"] for c in cases], selection=selection,
        basis_subjects=208, regressor_subjects=len(cases), validation_used=False, basis_rank=32,
        parameter_count=sum(p.numel() for p in model.parameters()), seed=0, augmentation="none",
        architecture="Conv3d 5->16->32->64->64, stride 2, GroupNorm/GELU; adaptive spatial pool 4^3; linear 4096->128->32; zero final layer",
        inputs=["MRI", "foreground probability", "clipped source SDF /4", "predicted inner band", "predicted outer band"],
        alignment="same prediction-centroid integer translation into 80^3 canvas as spectral analysis",
        objective="per-case oracle field MSE normalized by zero-coefficient oracle energy (floor .0001) + .05 whole-union soft Dice loss",
        coefficient_scaling="global per-mode RMS over union of selected training predicted bands; target-independent, fixed",
        optimizer=f"Adam lr .0003; cosine to .000003; batch {args.batch_size if args.all_training else 8}; clip gradient norm 5; FP32; no weight decay",
        steps=total_steps, epochs=args.epochs if args.all_training else args.steps,
        batch_size=args.batch_size if args.all_training else 8,
        shuffle="fresh seed-0 permutation each epoch" if args.all_training else "full batch; no shuffle",
        selection_metric=f"lowest training relative oracle-field MSE at evaluations every {evaluation_interval} steps; final also reported",
        diagnostic_success="at least 80% of mean oracle Dice gain AND >=95% reduction in normalized oracle-field squared error",
        torch_version=torch.__version__, gpu=torch.cuda.get_device_name(),
        caveats=["training results alone do not assess generalization; evaluate saved checkpoint separately", "PCA learned from 208 training annotations including selected subjects", "baseline predictions in-sample", "distance-field changes are not guaranteed invertible or topology preserving"],
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        shape_variance_explained=float(explained[:32].sum()))
    write_json(args.output / "protocol.json", protocol)
    torch.save(dict(modes=modes.cpu(), error_mean=error_mean.cpu(), common_scale=common_scale.cpu()), args.output / "fixed_basis.pt")
    history, best_error, best_step = [], float("inf"), 0
    fit_start = time.time()
    shuffle_rng = np.random.default_rng(0)
    order = np.arange(len(cases))
    for step in range(total_steps + 1):
        if step % evaluation_interval == 0 or step == total_steps:
            rows, summary = evaluate(model, inputs, cases, modes, common_scale, error_mean)
            row = dict(step=step, epoch=step/batches_per_epoch if args.all_training else step,
                       elapsed_seconds=time.time()-fit_start, **summary)
            history.append(row)
            if summary["relative_oracle_field_mse"] < best_error:
                best_error, best_step = summary["relative_oracle_field_mse"], step
                torch.save(dict(model=model.state_dict(), step=step), args.output / "best.pt")
            write_json(args.output / "history.json", history)
            print(json.dumps(row), flush=True)
        if step == total_steps:
            break
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if args.all_training:
            position = step % batches_per_epoch
            if position == 0:
                order = shuffle_rng.permutation(len(cases))
                assert np.array_equal(np.sort(order), np.arange(208))
            indices = order[position*args.batch_size:(position+1)*args.batch_size]
            batch_inputs = inputs[torch.as_tensor(indices, device=device)]
            batch_cases = [cases[int(i)] for i in indices]
        else:
            batch_inputs, batch_cases = inputs, cases
        coefficients = model(batch_inputs)
        loss, field_loss, boundary_loss = field_objective(coefficients, batch_cases)
        assert torch.isfinite(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
    torch.save(dict(model=model.state_dict(), step=total_steps), args.output / "last.pt")
    _, last_summary = evaluate(model, inputs, cases, modes, common_scale, error_mean)
    model.load_state_dict(torch.load(args.output / "best.pt", map_location=device, weights_only=False)["model"])
    saved = args.output / "cases"
    saved.mkdir()
    rows, summary = evaluate(model, inputs, cases, modes, common_scale, error_mean, saved)
    success = (summary["oracle_gain_recovered"] is not None and summary["oracle_gain_recovered"] >= .8 and
               summary["relative_oracle_field_mse"] <= .05)
    result = dict(best_step=best_step, steps=total_steps, summary=summary, last=last_summary,
                  diagnostic_success=success, cases=rows, total_seconds=time.time()-start,
                  training_seconds=time.time()-fit_start)
    write_json(args.output / "results.json", result)
    render(args.output, history, rows)
    print("COMPLETE", json.dumps(result), flush=True)


def render(root, history, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    axes[0].plot([r["step"] for r in history], [r["learned_dice"] for r in history])
    axes[0].axhline(history[0]["baseline_dice"], color="gray", linestyle="--", label="Baseline")
    axes[0].axhline(history[0]["oracle_dice"], color="green", linestyle="--", label="Oracle")
    axes[0].set(xlabel="Training steps", ylabel="Mean foreground Dice", title=f"{len(rows)} training cases")
    axes[0].legend()
    axes[1].semilogy([r["step"] for r in history], [max(r["relative_oracle_field_mse"], 1e-10) for r in history])
    axes[1].axhline(.05, color="gray", linestyle="--")
    axes[1].set(xlabel="Training steps", ylabel="Relative oracle-field squared error", title="Learning the correction fields")
    if len(rows) <= 16:
        x = np.arange(len(rows))
        for shift, name in ((-.25, "baseline"), (0, "oracle"), (.25, "learned")):
            axes[2].bar(x+shift, [r[f"{name}_dice"] for r in rows], width=.25, label=name)
        axes[2].set_xticks(x, [r["case"].split("_")[-1] for r in rows], rotation=45)
        axes[2].set(ylabel="Foreground Dice", ylim=(.8, 1), title="Per-case comparison")
        axes[2].legend(fontsize=8)
    else:
        axes[2].scatter([r["oracle_dice"]-r["baseline_dice"] for r in rows],
                        [r["learned_dice"]-r["baseline_dice"] for r in rows], s=12)
        axes[2].axhline(0, color="gray", linewidth=.7)
        axes[2].axvline(0, color="gray", linewidth=.7)
        axes[2].set(xlabel="Oracle Dice gain", ylabel="Learned Dice gain", title="Training gains per case")
    fig.savefig(root / "overview.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
