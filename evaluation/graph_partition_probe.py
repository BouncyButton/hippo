"""Training-only graph-interface screen; see its dated frozen protocol."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import scipy
import sklearn
from scipy import ndimage, sparse
from scipy.sparse.csgraph import breadth_first_order, maximum_flow
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.audit_ap_partition_topology import score
from evaluation.probe_context_increment import context_maps, digest, prediction_features, save_json
from evaluation.pilot_harmonic_ap import fixed_cut_labels, measure
from thesis.new_constraints.uncal_fold.audit_predicted_foreground import largest_foreground_component
from thesis.new_constraints.uncal_fold.foldedness import best_fit_first_anterior_slice

SEED = 20260923
ARMS = ("base", "existing_mri", "surface", "graph_context", "shuffled_graph")
LAMBDAS = (0., .05, .2, 1.)


def interface_features(image: np.ndarray, support: np.ndarray,
                       candidates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pool geometric and MRI signals on AP graph faces; no labels or cut target."""
    support = np.asarray(support, dtype=bool)
    if image.shape != support.shape or image.ndim != 3 or not support.any():
        raise ValueError("Expected matching nonempty 3D image/support")
    if not np.isfinite(image).all() or np.any(candidates < 1) or np.any(candidates >= image.shape[1]):
        raise ValueError("Invalid image or candidate")
    inside = ndimage.distance_transform_edt(support)
    sdf = inside - ndimage.distance_transform_edt(~support)
    geometry = [inside, (inside <= 1.5).astype(float)]
    for sigma in (1., 2.):
        smooth = ndimage.gaussian_filter(sdf, sigma)
        gradient = np.stack(np.gradient(smooth))
        normals = gradient / np.maximum(np.linalg.norm(gradient, axis=0), 1e-6)
        curvature = sum(np.gradient(normals[a], axis=a) for a in range(3))
        geometry.extend((normals[1], np.abs(normals[1]), curvature, np.abs(curvature)))
    geometry = np.stack(geometry)
    mri = context_maps(image, support)
    center = np.argwhere(support).mean(axis=0)
    x, z = np.indices((support.shape[0], support.shape[2]))
    sectors = [np.ones(x.shape, bool)] + [
        ((x >= center[0]) == high_x) & ((z >= center[2]) == high_z)
        for high_x in (False, True) for high_z in (False, True)]
    shape_rows, mri_rows = [], []
    for cut in candidates:
        valid = support[:, cut - 1, :] & support[:, cut, :]
        g = .5 * (geometry[:, :, cut - 1, :] + geometry[:, :, cut, :])
        left, right = mri[:, :, cut - 1, :], mri[:, :, cut, :]
        fields = np.concatenate((.5 * (left + right), np.abs(left - right)))
        sr, mr = [], []
        for sector in sectors:
            mask = valid & sector
            n = int(mask.sum())
            sr.extend((n / max(int(support.sum()), 1), float(n > 0)))
            for source, row in ((g, sr), (fields, mr)):
                if n:
                    values = source[:, mask]
                    row.extend(values.mean(axis=1))
                    row.extend(values.std(axis=1))
                else:
                    row.extend(np.zeros(2 * len(source)))
        shape_rows.append(sr)
        mri_rows.append(mr)
    result = tuple(np.asarray(rows, dtype=np.float32) for rows in (shape_rows, mri_rows))
    if not all(np.isfinite(a).all() for a in result):
        raise ValueError("Nonfinite interface features")
    return result


def graph_cut(margin: np.ndarray, prediction: np.ndarray, lam: float,
              scale: int = 10000) -> tuple[np.ndarray, dict]:
    """Six-face binary cut, source side anterior; exact for rounded capacities."""
    if margin.shape != prediction.shape or margin.ndim != 3 or not np.isfinite(margin).all():
        raise ValueError("Invalid margin/prediction")
    if not np.isin(prediction, (0, 1, 2)).all() or not np.isfinite(lam) or lam < 0 or scale <= 0:
        raise ValueError("Invalid labels or capacities")
    support = prediction > 0
    n = int(support.sum())
    indices = np.full(prediction.shape, -1, dtype=np.int64)
    indices[support] = np.arange(n)
    edge_i, edge_j = [], []
    for axis in range(3):
        a, b = [slice(None)] * 3, [slice(None)] * 3
        a[axis], b[axis] = slice(None, -1), slice(1, None)
        a, b = tuple(a), tuple(b)
        valid = support[a] & support[b]
        edge_i.append(indices[a][valid])
        edge_j.append(indices[b][valid])
    ei, ej = np.concatenate(edge_i), np.concatenate(edge_j)
    values = margin[support].astype(np.float64)
    cost_a, cost_p = np.logaddexp(0, -values), np.logaddexp(0, values)
    out = prediction.copy()
    if lam == 0 or n == 0:
        labels_a = values >= 0
    else:
        offset = np.minimum(cost_a, cost_p)
        # Source->node pays P cost, node->sink pays A cost.
        capacities = np.r_[cost_p - offset, cost_a - offset, np.full(2 * len(ei), lam)]
        rounded = np.rint(capacities * scale).astype(np.int64)
        if rounded.max(initial=0) >= np.iinfo(np.int32).max:
            raise ValueError("Capacity exceeds solver integer range")
        rows = np.r_[np.full(n, n), np.arange(n), ei, ej]
        cols = np.r_[np.arange(n), np.full(n, n + 1), ej, ei]
        capacity = sparse.csr_matrix((rounded, (rows, cols)), shape=(n + 2, n + 2))
        capacity.eliminate_zeros()
        flow = maximum_flow(capacity, n, n + 1)
        residual = (capacity - flow.flow).tocsr()
        residual.data = (residual.data > 0).astype(np.int64)
        residual.eliminate_zeros()
        reachable = breadth_first_order(residual, n, directed=True, return_predecessors=False)
        labels_a = np.zeros(n + 2, dtype=bool)
        labels_a[reachable] = True
        labels_a = labels_a[:n]
    out[support] = np.where(labels_a, 1, 2)
    objective = np.where(labels_a, cost_a, cost_p).sum() + lam * (labels_a[ei] != labels_a[ej]).sum()
    return out, {"energy": float(objective), "nodes": n, "edges": len(ei),
                 "single_class": bool(n and np.unique(out[support]).size == 1),
                 "float_optimality_gap_bound": (n + len(ei)) / scale if lam else 0.}


def synthetic_base(prediction, candidates, cut):
    margin = np.broadcast_to(2. * (np.arange(prediction.shape[1])[None, :, None] - cut + .5), prediction.shape)
    logits = np.stack((np.zeros_like(margin), .5 * margin, -.5 * margin))
    union, _, _ = largest_foreground_component(prediction > 0)
    return prediction_features(logits, prediction, union, candidates, cut)


def design(case, arm, task, variant=0):
    base = case["base"] if task == "natural" else case["synthetic_base"][variant]
    if arm == "base":
        return base
    if arm == "existing_mri":
        block = case["old_mri"]
    elif arm == "surface":
        block = case["surface"]
    elif arm in ("graph_context", "shuffled_graph"):
        block = np.column_stack((case["surface"], case["edge_mri"]))
        if arm == "shuffled_graph":
            seed = int.from_bytes(hashlib.sha256(str(case["name"].item()).encode()).digest()[:8], "little")
            block = np.random.default_rng(seed).permutation(block, axis=0)
    else:
        raise ValueError(arm)
    return np.column_stack((base, block))


def fit(cases, arm, task):
    xs, ys, weights = [], [], []
    count = 1 if task == "natural" else 3
    for case in cases:
        positive = case["candidates"] == int(case["target"])
        if positive.sum() != 1 or len(positive) < 2:
            raise ValueError("Target outside candidate support")
        for variant in range(count):
            xs.append(design(case, arm, task, variant))
            ys.append(positive)
            weights.append(np.where(positive, .5, .5 / (len(positive) - 1)) / count)
    estimator = make_pipeline(StandardScaler(), LogisticRegression(C=.1, solver="lbfgs", max_iter=3000, tol=1e-7))
    estimator.fit(np.concatenate(xs), np.concatenate(ys), logisticregression__sample_weight=np.concatenate(weights))
    if estimator[-1].n_iter_.max() >= 3000:
        raise RuntimeError("Readout did not converge")
    return estimator


def provenance(args, names):
    paths = [Path(__file__), args.output / "PROTOCOL.md", args.split,
             ROOT / "evaluation/probe_context_increment.py",
             ROOT / "evaluation/audit_ap_partition_topology.py",
             ROOT / "thesis/new_constraints/uncal_fold/foldedness.py",
             ROOT / "baselines/swin_unetr/swin_unetr.py"]
    return {"names": names, "files": {str(p.relative_to(ROOT)): digest(p) for p in paths},
            "versions": {"numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__},
            "old_manifest": json.loads((args.old_cache / "manifest.json").read_text())}


def extract(args, names):
    import torch
    from baselines.swin_unetr.swin_unetr import _build_monai_dataset_from_pkl, _load_pkl_dataframe
    from thesis.new_constraints.train_swinunetr_constraints import build_swinunetr
    torch.set_num_threads(4)
    args.cache.mkdir(parents=True, exist_ok=True)
    manifest = provenance(args, names)
    checkpoint = ROOT / "experiments/augmentation_family_b_20260907/checkpoints/baseline_seed0_checkpoint_best.pt"
    if digest(checkpoint) != manifest["old_manifest"]["checkpoint_sha256"]:
        raise ValueError("Checkpoint differs from compact cache")
    manifest["checkpoint_sha256"] = digest(checkpoint)
    pkl = ROOT / "datasets/Dataset101_MSD/msd_hippocampus_full.pkl"
    if digest(pkl) != manifest["old_manifest"]["pkl_sha256"]:
        raise ValueError("Dataset descriptor differs from compact cache")
    manifest["torch"] = torch.__version__
    path = args.cache / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Existing cache provenance differs")
    save_json(path, manifest)
    data = _build_monai_dataset_from_pkl(_load_pkl_dataframe(pkl), "MSD", 3, spatial_size=(64, 64, 64), do_resize=False)
    indices = {entry["case_name"]: i for i, entry in enumerate(data.data)}
    model = build_swinunetr((64, 64, 64), 3, torch.device("cpu"), activation_checkpointing=False)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state.get("run", {}).get("fold") != 0 or state.get("run", {}).get("seed") != 0:
        raise ValueError("Wrong model fold or seed")
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    t0 = time.monotonic()
    for i, name in enumerate(names):
        target_path = args.cache / f"{name}.npz"
        if target_path.exists():
            continue
        item = data[indices[name]]
        image, truth = item["image"][0].numpy(), item["label"][0].numpy().astype(np.uint8)
        with torch.inference_mode():
            logits = model(item["image"][None].float())[0].numpy()
        prediction = logits.argmax(axis=0).astype(np.uint8)
        union, _, _ = largest_foreground_component(prediction > 0)
        old_path = args.old_cache / f"{name}.npz"
        with np.load(old_path, allow_pickle=False) as old:
            old = dict(old)
        candidates = old["candidates"]
        target = best_fit_first_anterior_slice(truth)[0]
        original_cut = best_fit_first_anterior_slice(prediction)[0]
        base = prediction_features(logits, prediction, union, candidates, original_cut)
        if target != int(old["target"]) or original_cut != int(old["original_cut"]):
            raise ValueError(f"Cut/cache mismatch {name}")
        np.testing.assert_allclose(base, old["base"], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(score(prediction, truth)["macro_dice"], old["original_dice"], rtol=0, atol=1e-12)
        surface, edge_mri = interface_features(image, union, candidates)
        starts = np.array([target - 2, target, target + 2])
        if not np.isin(starts, candidates).all():
            raise ValueError(f"Synthetic displacement outside support {name}")
        synthetic = np.stack([synthetic_base(prediction, candidates, c) for c in starts])
        input_files = [ROOT / "datasets/Dataset101_MSD/imagesTr" / f"{name}_0000.nii.gz",
                       ROOT / "datasets/Dataset101_MSD/labelsTr" / f"{name}.nii.gz"]
        with target_path.with_suffix(".tmp").open("wb") as stream:
            np.savez_compressed(stream, name=name, base=base, old_mri=old["mri"],
                                candidates=candidates, target=target, original_cut=original_cut,
                                original_dice=old["original_dice"], original_swaps=old["original_swaps"],
                                cut_dice=old["cut_dice"], cut_swaps=old["cut_swaps"],
                                surface=surface, edge_mri=edge_mri, synthetic_base=synthetic,
                                starts=starts, prediction=prediction, truth=truth,
                                margin=logits[1] - logits[2], old_cache_sha256=digest(old_path),
                                native_hashes=np.array([digest(p) for p in input_files]))
        target_path.with_suffix(".tmp").replace(target_path)
        if (i + 1) % 8 == 0 or i == len(names) - 1:
            print(f"extract {i+1}/{len(names)} ({time.monotonic()-t0:.0f}s)", flush=True)


def paired(values):
    values = np.asarray(values, dtype=float)
    draws = np.random.default_rng(SEED).integers(len(values), size=(10000, len(values)))
    return {"mean": float(values.mean()), "ci95": np.quantile(values[draws].mean(axis=1), [.025, .975]).tolist(),
            "negative": int((values < -1e-12).sum()), "zero": int((np.abs(values) <= 1e-12).sum()),
            "positive": int((values > 1e-12).sum())}


def cut_metrics(case, cut):
    k = np.flatnonzero(case["candidates"] == cut)
    if len(k) != 1:
        raise ValueError("Cut outside support")
    return {"cut": int(cut), "mae": abs(int(cut) - int(case["target"])),
            "dice": float(case["cut_dice"][k[0]]), "swaps": int(case["cut_swaps"][k[0]])}


def summarize_rows(rows, methods):
    result = {"n": len(rows), "methods": {}}
    for method in methods:
        result["methods"][method] = {key: float(np.mean([row[method][key] for row in rows])) for key in ("mae", "dice")}
        result["methods"][method]["swaps"] = int(sum(row[method]["swaps"] for row in rows))
    result["comparisons"] = {}
    for method in methods:
        for control in ("original", "base", "existing_mri", "shuffled_graph"):
            if control not in methods or method == control:
                continue
            result["comparisons"][f"{method}_minus_{control}"] = {
                "mae": paired([r[method]["mae"] - r[control]["mae"] for r in rows]),
                "dice_pp": paired([100 * (r[method]["dice"] - r[control]["dice"]) for r in rows])}
    return result


def screen(args, names):
    manifest = json.loads((args.cache / "manifest.json").read_text())
    current = provenance(args, names)
    if any(manifest[key] != value for key, value in current.items()):
        raise ValueError("Protocol/source/cache manifest changed after extraction")
    cases, hashes = [], {}
    for name in names:
        path = args.cache / f"{name}.npz"
        hashes[name] = digest(path)
        with np.load(path, allow_pickle=False) as data:
            case = dict(data)
        if digest(args.old_cache / f"{name}.npz") != str(case["old_cache_sha256"]):
            raise ValueError("Compact input cache changed")
        cases.append(case)
    folds = list(KFold(4, shuffle=True, random_state=SEED).split(cases))
    all_rows, summaries = {}, {}
    for task in ("natural", "displaced"):
        rows = []
        for fold, (train, held) in enumerate(folds):
            estimators = {arm: fit([cases[i] for i in train], arm, task) for arm in ARMS}
            for i in held:
                case = cases[i]
                for variant in range(1 if task == "natural" else 3):
                    start = int(case["original_cut"]) if task == "natural" else int(case["starts"][variant])
                    row = {"name": str(case["name"].item()), "fold": fold,
                           "displacement": 0 if task == "natural" else (-2, 0, 2)[variant],
                           "target": int(case["target"]), "plane": cut_metrics(case, start)}
                    row["original"] = ({"cut": start, "mae": abs(start - row["target"]),
                                        "dice": float(case["original_dice"]), "swaps": int(case["original_swaps"])}
                                       if task == "natural" else dict(row["plane"]))
                    for arm, estimator in estimators.items():
                        scores = estimator.decision_function(design(case, arm, task, variant))
                        cut = case["candidates"][scores.argmax()]
                        row[arm] = cut_metrics(case, cut)
                    rows.append(row)
            print(f"{task} fold {fold+1}/4 complete", flush=True)
        all_rows[task] = rows
        methods = ("original", "plane", *ARMS)
        if task == "natural":
            summaries[task] = summarize_rows(rows, methods)
            summaries[task]["folds"] = {str(f): summarize_rows([r for r in rows if r["fold"] == f], methods)["methods"] for f in range(4)}
        else:
            summaries[task] = {str(d): summarize_rows([r for r in rows if r["displacement"] == d], methods) for d in (-2, 0, 2)}
    save_json(args.output / "descriptor_cases.json", all_rows)
    save_json(args.output / "descriptor_summary.json", summaries)
    # Independent uniform-affinity intervention, with training-only fold selection.
    graph_rows = []
    for i, case in enumerate(cases):
        row = {"name": str(case["name"].item()), "original": {
            "mae": abs(int(case["original_cut"]) - int(case["target"])),
            "dice": float(case["original_dice"]), "swaps": int(case["original_swaps"])}}
        for lam in LAMBDAS:
            corrected, diagnostics = graph_cut(case["margin"], case["prediction"], lam)
            if not np.array_equal(corrected > 0, case["prediction"] > 0):
                raise AssertionError("Foreground changed")
            if lam == 0:
                np.testing.assert_array_equal(corrected, case["prediction"])
            result = measure(corrected, case["truth"], int(case["target"]))
            row[str(lam)] = {"mae": result["cut_mae"], "dice": result["macro_dice"],
                             "swaps": result["ap_swaps"], **diagnostics}
        graph_rows.append(row)
        if (i+1) % 26 == 0:
            print(f"graph cuts {i+1}/{len(cases)}", flush=True)
    selections = []
    for fold, (train, held) in enumerate(folds):
        chosen = max(LAMBDAS, key=lambda lam: (np.mean([graph_rows[i][str(lam)]["dice"] for i in train]), -lam))
        selections.append({"fold": fold, "lambda": chosen})
        for i in held:
            graph_rows[i]["selected"] = graph_rows[i][str(chosen)]
            graph_rows[i]["fold"] = fold
    graph_summary = summarize_rows(graph_rows, ("original", "selected", *(str(v) for v in LAMBDAS)))
    graph_summary["selections"] = selections
    graph_summary["single_class_cases"] = {str(l): sum(r[str(l)]["single_class"] for r in graph_rows) for l in LAMBDAS}
    natural = summaries["natural"]
    comparisons = natural["comparisons"]
    checks = {
        "natural_mae_gain_0.1_vs_base": comparisons["graph_context_minus_base"]["mae"]["mean"] <= -.1,
        "natural_mae_gain_0.1_vs_existing_mri": comparisons["graph_context_minus_existing_mri"]["mae"]["mean"] <= -.1,
        "natural_mae_interval_below_zero": comparisons["graph_context_minus_base"]["mae"]["ci95"][1] < 0,
        "natural_dice_not_worse": comparisons["graph_context_minus_base"]["dice_pp"]["mean"] >= 0,
        "beats_shuffle": comparisons["graph_context_minus_shuffled_graph"]["mae"]["mean"] < 0,
    }
    for d in (-2, 0, 2):
        delta = summaries["displaced"][str(d)]["comparisons"]["graph_context_minus_base"]["mae"]["mean"]
        checks[f"displacement_{d}"] = delta < 0 if d else delta <= 0
    gc = graph_summary["comparisons"]["selected_minus_original"]
    graph_checks = {"dice_gain_0.1pp": gc["dice_pp"]["mean"] >= .1,
                    "dice_interval_positive": gc["dice_pp"]["ci95"][0] > 0,
                    "cut_mae_improves": gc["mae"]["mean"] < 0,
                    "no_more_harmed_than_helped": gc["dice_pp"]["negative"] <= gc["dice_pp"]["positive"]}
    save_json(args.output / "graph_cases.json", graph_rows)
    save_json(args.output / "graph_summary.json", graph_summary)
    save_json(args.output / "decision.json", {"descriptor_checks": checks, "descriptor_advance": all(checks.values()),
                                              "graph_checks": graph_checks, "graph_advance": all(graph_checks.values())})
    save_json(args.output / "manifest.json", {**manifest, "cache_hashes": hashes,
                                             "interpretation": "Training-only exploratory case holdout; in-sample backbone; no development evaluation."})
    print(json.dumps({"descriptor": checks, "graph": graph_checks}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("extract", "screen"), required=True)
    parser.add_argument("--cache", type=Path, default=ROOT / "experiments/graph_partition_20260923/cache")
    parser.add_argument("--old-cache", type=Path, default=ROOT / "experiments/context_increment_20260923/cache/unaugmented")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/experiments/graph_partition_20260923")
    parser.add_argument("--split", type=Path, default=ROOT / "datasets/Dataset101_MSD/splits_final.json")
    args = parser.parse_args()
    split = json.loads(args.split.read_text())[0]
    names = sorted(split["train"])
    if set(names) & set(split["val"]):
        raise ValueError("Overlapping split")
    args.output.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=2):
        (extract if args.stage == "extract" else screen)(args, names)


if __name__ == "__main__":
    main()
