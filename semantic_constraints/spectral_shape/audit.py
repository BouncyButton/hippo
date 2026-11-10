"""Train-only shape/error PCA and fixed DCT boundary correction oracle audit.

No neural network is trained. Validation annotations are used only for explicitly
labelled oracle coefficient fitting, reconstruction analysis, and evaluation.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
from scipy import fft, ndimage as ndi
import torch

from semantic_constraints.spline_refinement.train import prepare, signed_distance, dice, surface_distances, write_json

SIZE = 80
RANKS = (8, 32, 128)
RIDGE = .001


def shift_into_canvas(array, shift, size=SIZE):
    """Integer translation, zero padding, no resampling or anatomical scaling."""
    out = np.zeros((size,) * 3, dtype=array.dtype)
    source, dest = [], []
    for n, delta in zip(array.shape, shift):
        lo, hi = max(0, -int(delta)), min(n, size - int(delta))
        if hi <= lo:
            raise ValueError("No canvas overlap")
        source.append(slice(lo, hi))
        dest.append(slice(lo + int(delta), hi + int(delta)))
    out[tuple(dest)] = array[tuple(source)]
    assert np.count_nonzero(out) == np.count_nonzero(array), "Translation cropped foreground"
    return out


def bands(mask):
    return ndi.binary_dilation(mask, iterations=2) & ~ndi.binary_erosion(mask, iterations=2)


def fit_pca(values, rank=128):
    """Dual PCA, float64 Gram eigensystem; values [subjects, spatial samples]."""
    mean = values.mean(0)
    centered = values - mean
    gram = centered.double() @ centered.double().T
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues, eigenvectors = eigenvalues[order].clamp_min(0), eigenvectors[:, order]
    positive = eigenvalues > eigenvalues[0] * 1e-9
    count = min(rank, int(positive.sum()))
    modes = (eigenvectors[:, :count].T.float() @ centered) / eigenvalues[:count].sqrt().float()[:, None]
    explained = eigenvalues / eigenvalues.sum()
    return mean, modes, explained


def dct_frequencies(size=SIZE, count=512):
    # Include complete candidates beyond the radial cutoff, then break ties
    # lexicographically. No data or validation information chooses frequencies.
    frequencies = np.indices((size,) * 3).reshape(3, -1).T
    order = np.lexsort((frequencies[:, 2], frequencies[:, 1], frequencies[:, 0], (frequencies ** 2).sum(1)))
    return frequencies[order[:count]]


def dct_design(points, frequencies, size=SIZE):
    points = points.to(dtype=torch.float32)
    frequencies = frequencies.to(dtype=torch.float32)
    design = torch.ones((len(points), len(frequencies)), device=points.device)
    for axis in range(3):
        frequency = frequencies[None, :, axis]
        normalization = torch.where(frequency == 0, 1 / np.sqrt(size), np.sqrt(2 / size))
        design *= normalization * torch.cos(torch.pi * (points[:, axis, None] + .5) * frequency / size)
    return design


def band_fit(design, target, mean, ridge=RIDGE):
    """Band least-squares oracle; standardize basis columns before fixed ridge."""
    scale = design.square().mean(0).sqrt().clamp_min(1e-8)
    normalized = design / scale
    n = len(target)
    gram = normalized.T @ normalized / n
    rhs = normalized.T @ (target - mean) / n
    coefficients = torch.linalg.solve(gram + ridge * torch.eye(gram.shape[0], device=gram.device), rhs)
    fitted = mean + normalized @ coefficients
    assert torch.isfinite(fitted).all()
    return fitted, coefficients / scale


def summarize(rows):
    groups = {}
    for method in sorted({r["method"] for r in rows}):
        items = [r for r in rows if r["method"] == method]
        delta = np.array([r["dice_gain"] for r in items])
        bootstrap = np.random.default_rng(0).choice(delta, size=(2000, len(delta))).mean(1)
        groups[method] = dict(cases=len(items), mean_dice=float(np.mean([r["dice"] for r in items])),
            mean_dice_gain=float(delta.mean()), median_dice_gain=float(np.median(delta)),
            descriptive_gain_ci95=np.quantile(bootstrap, [.025, .975]).tolist(),
            improved=int((delta > 1e-10).sum()), worsened=int((delta < -1e-10).sum()),
            unchanged=int((abs(delta) <= 1e-10).sum()),
            mean_assd_mm=float(np.mean([r["assd_mm"] for r in items])),
            mean_hd95_mm=float(np.mean([r["hd95_mm"] for r in items])),
            fixed=sum(r["fixed"] for r in items), introduced=sum(r["introduced"] for r in items),
            total_fp=sum(r["fp"] for r in items), total_fn=sum(r["fn"] for r in items),
            outside_band_changes=sum(r["outside_band_changes"] for r in items),
            topology_changed_cases=sum(r["components_changed"] for r in items))
        if "band_mse" in items[0]:
            groups[method]["mean_band_mse"] = float(np.mean([r["band_mse"] for r in items]))
    return groups


def evaluate_mask(case, mask, method, correction=None):
    source, truth, support = case["prediction"], case["truth"], case["support"]
    assert np.array_equal(mask[~support], source[~support]), method
    score = dice(mask, truth)
    assd, hd95 = surface_distances(mask, truth)
    baseline_score = dice(source, truth)
    row = dict(case=case["name"], method=method, dice=score, baseline_dice=baseline_score,
               dice_gain=score - baseline_score, assd_mm=assd, hd95_mm=hd95,
               fp=int((mask & ~truth).sum()), fn=int((~mask & truth).sum()),
               fixed=int(((source != truth) & (mask == truth)).sum()),
               introduced=int(((source == truth) & (mask != truth)).sum()),
               outside_band_changes=int(((source != mask) & ~support).sum()),
               components_changed=ndi.label(source)[1] != ndi.label(mask)[1])
    if correction is not None:
        row["band_mse"] = float(np.mean((correction - case["residual"][support]) ** 2))
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("checkpoint", "pkl", "splits", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    args = parser.parse_args()
    args.smoke = False
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "cases").mkdir()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")
    start = time.time()
    records, provenance = prepare(args, device)
    cases = {"train": [], "val": []}
    prediction_digest = hashlib.sha256()
    for cohort in cases:
        for record in records[cohort]:
            prediction = record["prediction"].numpy() > 0
            labels = record["labels"].numpy()
            shift = np.rint((SIZE - 1) / 2 - np.argwhere(prediction).mean(0)).astype(int)
            p = shift_into_canvas(prediction, shift)
            g = shift_into_canvas(labels > 0, shift)
            pred_sdf = np.clip(signed_distance(p), -8, 8)
            gt_sdf = np.clip(signed_distance(g), -8, 8)
            support = bands(p)
            # Check independent SciPy morphology against the established torch bands.
            original_support = shift_into_canvas(record["support"][0].numpy().astype(bool), shift)
            assert np.array_equal(original_support, support)
            cases[cohort].append(dict(name=record["name"], prediction=p, truth=g,
                support=support, source_sdf=pred_sdf, target_sdf=gt_sdf,
                residual=gt_sdf - pred_sdf, shift=shift.tolist()))
            if cohort == "val":
                prediction_digest.update(record["name"].encode())
                prediction_digest.update(record["prediction"].numpy().tobytes())
    del records
    torch.cuda.empty_cache()
    protocol = dict(**provenance, purpose="Representation analysis; validation oracle fits are not predictions",
        canvas=[SIZE] * 3, spacing_mm=1, signed_distance_clip_mm=8,
        alignment="Prediction-centroid integer translation only; common dataset axes, no rotation/reflection/scale fitting; same shift for prediction and truth",
        rank_grid=list(RANKS), dct_extra_rank=512, pca_fit_subjects="208 training subjects only",
        pca_error_target="full-canvas clipped truth SDF minus clipped predicted SDF, before band gating",
        pca_shape_target="clipped annotation SDF", coefficient_fit="oracle band least squares with column RMS normalization and fixed ridge .001",
        pca_mean="Training mean error is a fixed offset in every oracle correction family",
        controls=["identity", "mean_training_error_only", "projection_of_prediction_onto_training_shape_PCA", "band_ceiling"],
        masks="Edits restricted to prediction-derived 2-step six-connected bands; foreground union only",
        interpretation="No neural regressor; no diffeomorphic or topology guarantee; record connected components",
        caveats=["Training predictions are in-sample", "fold0 validation is development data", "retained training pose variation can affect PCA", "PCA of distance fields need not yield valid signed-distance fields; only zero level is used", "rank grids reported in full, no selected validation winner"],
        validation_prediction_digest=prediction_digest.hexdigest(), torch_version=torch.__version__,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_json(args.output / "protocol.json", protocol)
    print("PREPARED", {c: len(v) for c, v in cases.items()}, flush=True)
    train_shapes = torch.from_numpy(np.stack([c["target_sdf"].ravel() for c in cases["train"]])).to(device)
    train_errors = torch.from_numpy(np.stack([c["residual"].ravel() for c in cases["train"]])).to(device)
    shape_mean, shape_modes, shape_variance = fit_pca(train_shapes)
    error_mean, error_modes, error_variance = fit_pca(train_errors)
    shape_sd = ((train_shapes - shape_mean) @ shape_modes[:8].T).std(0)
    error_sd = ((train_errors - error_mean) @ error_modes[:8].T).std(0)
    # Check orthonormality of the fitted basis, independently of reconstruction.
    checks = {}
    for name, modes in (("shape", shape_modes), ("error", error_modes)):
        error = float((modes @ modes.T - torch.eye(len(modes), device=device)).abs().max())
        assert error < 2e-4, error
        checks[f"{name}_orthonormal_max_error"] = error
    spectra = dict(shape_variance=shape_variance.cpu().tolist(), error_variance=error_variance.cpu().tolist(),
                   shape_training_cumulative={str(k): float(shape_variance[:k].sum()) for k in RANKS},
                   error_training_cumulative={str(k): float(error_variance[:k].sum()) for k in RANKS})
    write_json(args.output / "spectra.json", spectra)
    print("PCA", json.dumps({k: v for k, v in spectra.items() if "cumulative" in k}), flush=True)
    del train_shapes, train_errors
    frequencies_np = dct_frequencies()
    frequencies = torch.from_numpy(frequencies_np).to(device)
    np.savez_compressed(args.output / "modes_first8.npz",
        shape_mean=shape_mean.cpu().numpy().reshape((SIZE,) * 3),
        error_mean=error_mean.cpu().numpy().reshape((SIZE,) * 3),
        shape_modes=shape_modes[:8].cpu().numpy().reshape((-1, SIZE, SIZE, SIZE)),
        error_modes=error_modes[:8].cpu().numpy().reshape((-1, SIZE, SIZE, SIZE)),
        shape_sd=shape_sd.cpu().numpy(), error_sd=error_sd.cpu().numpy())
    rows, reconstruction_rows, energy_rows, mode_properties = [], [], [], []
    for cohort in ("train", "val"):
        for case_index, case in enumerate(cases[cohort]):
            shape = torch.from_numpy(case["target_sdf"].ravel()).to(device)
            error = torch.from_numpy(case["residual"].ravel()).to(device)
            pred = torch.from_numpy(case["source_sdf"].ravel()).to(device)
            support = case["support"]
            positions = np.flatnonzero(support)
            idx = torch.from_numpy(positions).to(device)
            target = error[idx]
            offset = error_mean[idx]
            source_band = pred[idx]
            shape_scores = shape_modes @ (shape - shape_mean)
            prediction_scores = shape_modes @ (pred - shape_mean)
            error_scores = error_modes @ (error - error_mean)
            coordinates = np.argwhere(case["truth"])
            extents = coordinates.max(0) - coordinates.min(0) + 1
            mode_properties.append(dict(case=case["name"], cohort=cohort,
                shape_scores=shape_scores[:8].cpu().tolist(), error_scores=error_scores[:8].cpu().tolist(),
                volume_voxels=int(case["truth"].sum()),
                prediction_volume_bias=int(case["prediction"].sum()) - int(case["truth"].sum()),
                **{f"extent_{a}_mm": int(extents[a]) for a in range(3)}))
            # Frequency energy is computed BEFORE band masking, which would
            # itself introduce high frequencies at the support boundary.
            coefficients = fft.dctn(case["residual"], type=2, norm="ortho")
            energy_total = float(np.square(coefficients, dtype=np.float64).sum())
            ordered_energy = np.square(coefficients[tuple(frequencies_np.T)], dtype=np.float64)
            energy_rows.append(dict(case=case["name"], cohort=cohort, total_residual_energy=energy_total,
                fractions={str(k): float(ordered_energy[:k].sum() / max(energy_total, 1e-12)) for k in (*RANKS, 512)}))
            for rank in RANKS:
                reconstructed_shape = shape_mean + shape_scores[:rank] @ shape_modes[:rank]
                reconstructed_error = error_mean + error_scores[:rank] @ error_modes[:rank]
                reconstruction_rows.append(dict(case=case["name"], cohort=cohort, rank=rank,
                    annotation_reconstruction_dice=dice((reconstructed_shape.cpu().numpy().reshape((SIZE,) * 3) < 0), case["truth"]),
                    annotation_sdf_mse=float((shape - reconstructed_shape).square().mean()),
                    error_centered_variance_captured=1-float((error-reconstructed_error).square().sum()/((error-error_mean).square().sum().clamp_min(1e-12)))))
            # Training reconstructions and spectra diagnose fit; only validation
            # receives expensive per-case band oracle fitting and surface metrics.
            if cohort == "train":
                continue
            baseline = evaluate_mask(case, case["prediction"], "identity", np.zeros(int(support.sum())))
            rows.append(baseline)
            saved = dict(prediction=case["prediction"], truth=case["truth"], support=support,
                         shift=np.array(case["shift"]), source_sdf=case["source_sdf"])

            def add_result(name, delta):
                delta_np = delta.detach().cpu().numpy()
                mask = case["prediction"].copy()
                mask[support] = case["source_sdf"][support] + delta_np < 0
                rows.append(evaluate_mask(case, mask, name, delta_np))
                if case["name"].endswith("164"):
                    saved[name] = mask

            add_result("mean_error_no_annotation", offset)
            add_result("band_ceiling_oracle", target)
            designs = {
                "error_pca": error_modes[:, idx].T,
                "shape_pca": shape_modes[:, idx].T,
                "dct": dct_design(torch.from_numpy(np.argwhere(support)).to(device), frequencies),
            }
            for rank in RANKS:
                # Actual annotation-free shape-prior baseline: project the
                # PREDICTED distance field onto the training annotation basis.
                projected_prediction = shape_mean[idx] + prediction_scores[:rank] @ shape_modes[:rank, idx]
                add_result(f"shape_projection_no_annotation_k{rank}", projected_prediction - source_band)
            for family, design in designs.items():
                for rank in (*RANKS, 512) if family == "dct" else RANKS:
                    fitted, _ = band_fit(design[:, :rank], target, offset)
                    add_result(f"{family}_oracle_k{rank}", fitted)
            # Preserve one explicitly specified family/rank for independent
            # mask verification on every validation case; not a selected winner.
            fitted, _ = band_fit(designs["error_pca"][:, :32], target, offset)
            verified_mask = case["prediction"].copy()
            verified_mask[support] = case["source_sdf"][support] + fitted.cpu().numpy() < 0
            saved["error_pca_oracle_k32"] = verified_mask
            np.savez_compressed(args.output / "cases" / f"{case['name']}.npz", **saved)
            if (case_index + 1) % 10 == 0:
                print(f"VALIDATION {case_index + 1}/52 elapsed={time.time()-start:.1f}s", flush=True)
    write_json(args.output / "case_metrics.json", rows)
    write_json(args.output / "reconstruction.json", reconstruction_rows)
    write_json(args.output / "frequency_energy.json", energy_rows)
    write_json(args.output / "mode_properties.json", mode_properties)
    correlations = {}
    training_properties = [r for r in mode_properties if r["cohort"] == "train"]
    for family in ("shape", "error"):
        correlations[family] = []
        for mode in range(8):
            score = [r[f"{family}_scores"][mode] for r in training_properties]
            correlations[family].append({key: float(np.corrcoef(score, [r[key] for r in training_properties])[0, 1])
                for key in ("volume_voxels", "prediction_volume_bias", "extent_0_mm", "extent_1_mm", "extent_2_mm")})
    summary = summarize(rows)
    reconstruction_summary = {}
    for cohort in ("train", "val"):
        reconstruction_summary[cohort] = {}
        for rank in RANKS:
            items = [r for r in reconstruction_rows if r["cohort"] == cohort and r["rank"] == rank]
            reconstruction_summary[cohort][str(rank)] = {
                key: float(np.mean([r[key] for r in items])) for key in
                ("annotation_reconstruction_dice", "annotation_sdf_mse", "error_centered_variance_captured")}
    result = dict(methods=summary, reconstruction=reconstruction_summary,
        frequency_energy={c: {str(k): float(np.mean([r["fractions"][str(k)] for r in energy_rows if r["cohort"] == c]))
                             for k in (*RANKS, 512)} for c in ("train", "val")},
        checks=checks, training_mode_correlations=correlations, elapsed_seconds=time.time()-start,
        case_164=[r for r in rows if r["case"].endswith("164")])
    write_json(args.output / "results.json", result)
    render(args.output, result, spectra)
    print("COMPLETE", json.dumps(result), flush=True)


def render(output, result, spectra):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for key, label in (("shape_variance", "Annotated shapes"), ("error_variance", "Prediction errors")):
        axes[0, 0].plot(np.arange(1, 209), np.cumsum(spectra[key]), label=label)
    axes[0, 0].set(title="PCA fitted on 208 training cases", xlabel="Number of modes", ylabel="Cumulative variance")
    axes[0, 0].legend()
    for family in ("error_pca", "shape_pca", "dct"):
        ranks = (*RANKS, 512) if family == "dct" else RANKS
        axes[0, 1].plot(ranks, [result["methods"][f"{family}_oracle_k{k}"]["mean_dice"] for k in ranks], "o-", label=family)
    axes[0, 1].axhline(result["methods"]["identity"]["mean_dice"], color="gray", linestyle="--", label="Baseline")
    axes[0, 1].set(xscale="log", xlabel="Number of fitted coefficients", ylabel="Validation Dice",
                   title="ANNOTATION-GUIDED fits: capacity, not predictions")
    axes[0, 1].legend(fontsize=8)
    for cohort in ("train", "val"):
        axes[1, 0].plot(RANKS, [result["reconstruction"][cohort][str(k)]["annotation_reconstruction_dice"] for k in RANKS], "o-", label=cohort)
    axes[1, 0].set(xscale="log", xlabel="Shape PCA modes", ylabel="Reconstructed annotation Dice", title="How compactly can anatomy be represented?")
    axes[1, 0].legend()
    names = ["mean_error_no_annotation"] + [f"shape_projection_no_annotation_k{k}" for k in RANKS]
    gains = [result["methods"][name]["mean_dice_gain"] * 100 for name in names]
    axes[1, 1].bar(["Mean error", "Shape 8", "Shape 32", "Shape 128"], gains, color=["seagreen" if g >= 0 else "indianred" for g in gains])
    axes[1, 1].set(ylabel="Validation Dice change, percentage points", title="Controls that DO NOT see validation annotations")
    fig.suptitle("Hippocampal shape and error modes: fold 0 exploratory analysis")
    fig.tight_layout()
    fig.savefig(output / "overview.png", dpi=150)
    plt.close(fig)
    modes = np.load(output / "modes_first8.npz")
    average = modes["shape_mean"]
    z = int(np.argmax((average < 0).sum((0, 1))))
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for i in range(3):
        mode = modes["shape_modes"][i] * modes["shape_sd"][i]
        for sign, color, style in ((-2, "royalblue", "-"), (0, "black", "--"), (2, "darkorange", "-")):
            axes[0, i].contour((average + sign * mode)[:, :, z].T, levels=[0], colors=[color], linestyles=[style])
        axes[0, i].set_title(f"Anatomical mode {i+1}: mean +/-2 SD")
        pattern = modes["error_modes"][i] * modes["error_sd"][i]
        limit = max(float(abs(pattern[:, :, z]).max()), .001)
        im = axes[1, i].imshow(pattern[:, :, z].T, origin="lower", cmap="coolwarm", vmin=-limit, vmax=limit)
        axes[1, i].contour(average[:, :, z].T, levels=[0], colors=["black"], linewidths=.8)
        axes[1, i].set_title(f"Error mode {i+1}: signed-distance change")
        fig.colorbar(im, ax=axes[1, i], label="mm per 1 SD")
        for ax in axes[:, i]:
            ax.set(xlim=(10, 70), ylim=(10, 70), aspect="equal")
    fig.suptitle(f"Training-derived modes, canvas slice {z}; blue=-2 SD, orange=+2 SD; signs arbitrary")
    fig.tight_layout()
    fig.savefig(output / "modes.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
