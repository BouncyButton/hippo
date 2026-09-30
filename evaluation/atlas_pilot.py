"""Calibrate, lock, and assess atlas-guided frozen-model refinement."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys

import numpy as np
from scipy import ndimage as ndi, sparse
from scipy.sparse.csgraph import maximum_flow, breadth_first_order

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.atlas_registration import CACHE, REPORT, native, normalize, save_json, sha
from evaluation.audit_ap_partition_topology import plane_projection

ALPHAS = (0., .1, .3, 1., 3.)
LAMBDAS = (0., .1, .3)
WHOLE = {
    "graph_only": [("deformable", 0., l) for l in LAMBDAS],
    "centered_fusion": [("centered", a, 0.) for a in ALPHAS],
    "affine_fusion": [("affine", a, 0.) for a in ALPHAS],
    "deformable_fusion": [("deformable", a, 0.) for a in ALPHAS],
    "deformable_graph": [("deformable", a, l) for a, l in itertools.product(ALPHAS, LAMBDAS)],
}
AP = {f"{kind}_ap": [(kind, a) for a in ALPHAS] for kind in ("centered", "affine", "deformable")}


def lattice_edges(image):
    ids = np.arange(image.size).reshape(image.shape)
    edges, weights = [], []
    for axis in range(3):
        lo, hi = [slice(None)] * 3, [slice(None)] * 3
        lo[axis], hi[axis] = slice(None, -1), slice(1, None)
        lo, hi = tuple(lo), tuple(hi)
        edges.append(np.column_stack((ids[lo].ravel(), ids[hi].ravel())))
        weights.append(np.exp(-np.square(image[lo] - image[hi]).ravel() / (.02)))
    return np.concatenate(edges), np.concatenate(weights)


def binary_cut(margin, edges, weights, lam, scale=10000):
    """Source side is foreground. Exact cut for the rounded binary energy."""
    shape = margin.shape
    margin = np.asarray(margin, dtype=np.float64).ravel()
    if not np.isfinite(margin).all() or not np.isfinite(weights).all() or lam < 0 or np.any(weights < 0):
        raise ValueError("Nonfinite costs or negative pairwise weights")
    if lam == 0:
        return (margin > 0).reshape(shape)
    n = len(margin)
    ei, ej = edges.T
    caps = np.r_[np.maximum(margin, 0), np.maximum(-margin, 0), lam * weights, lam * weights]
    rounded = np.rint(caps * scale).astype(np.int64)
    if rounded.max(initial=0) >= np.iinfo(np.int32).max:
        raise ValueError("Integer capacity overflow")
    row = np.r_[np.full(n, n), np.arange(n), ei, ej]
    col = np.r_[np.arange(n), np.full(n, n + 1), ej, ei]
    capacity = sparse.csr_matrix((rounded, (row, col)), shape=(n + 2, n + 2))
    capacity.eliminate_zeros()
    flow = maximum_flow(capacity, n, n + 1)
    residual = (capacity - flow.flow).tocsr()
    residual.data = (residual.data > 0).astype(np.int64)
    residual.eliminate_zeros()
    reachable = breadth_first_order(residual, n, directed=True, return_predecessors=False)
    foreground = np.zeros(n + 2, bool)
    foreground[reachable] = True
    return foreground[:n].reshape(shape)


def logit(probability):
    p = np.clip(probability, .01, .99)
    return np.log(p) - np.log1p(-p)


def whole_refine(logits, prior, edges, weights, alpha, lam):
    margin = np.maximum(logits[1], logits[2]) - logits[0]
    if alpha:
        margin = margin + alpha * logit(prior[1:].sum(0))
    foreground = binary_cut(margin, edges, weights, lam)
    return np.where(foreground, np.where(logits[1] >= logits[2], 1, 2), 0).astype(np.uint8)


def plane_from_margin(margin, support):
    occupied = np.flatnonzero(support.any(axis=(0, 2)))
    if len(occupied) < 2:
        raise ValueError("Plane requires at least two occupied slices")
    cuts = np.arange(occupied[0] + 1, occupied[-1] + 1)
    pooled = np.sum(np.where(support, margin, 0), axis=(0, 2))
    costs = np.r_[0., pooled.cumsum()][cuts]
    cut = int(cuts[np.argmin(costs)])
    labels = np.where(np.arange(support.shape[1])[None, :, None] >= cut, 1, 2)
    return np.where(support, labels, 0).astype(np.uint8), cut


def ap_refine(logits, prior, alpha, atlas_only=False):
    support = logits.argmax(0) > 0
    atlas_margin = np.log(np.clip(prior[1], .01, 1)) - np.log(np.clip(prior[2], .01, 1))
    margin = atlas_margin if atlas_only else logits[1] - logits[2] + alpha * atlas_margin
    return plane_from_margin(margin, support)


def dice(pred, target):
    den = int(pred.sum() + target.sum())
    return float(2 * (pred & target).sum() / den) if den else 1.


def measure(prediction, truth, raw, explicit_cut=None):
    fg, gt = prediction > 0, truth > 0
    raw_fg = raw > 0
    structure = ndi.generate_binary_structure(3, 1)
    a = fg & ~ndi.binary_erosion(fg, structure=structure, border_value=0)
    b = gt & ~ndi.binary_erosion(gt, structure=structure, border_value=0)
    if not a.any() or not b.any():
        assd = hd95 = float(np.linalg.norm(truth.shape))
    else:
        da = ndi.distance_transform_edt(~b)[a]
        db = ndi.distance_transform_edt(~a)[b]
        assd = float((da.mean() + db.mean()) / 2)
        hd95 = float(np.quantile(np.r_[da, db], .95))
    _, reference_plane = plane_projection(truth)
    _, predicted_plane = plane_projection(prediction)
    cut = predicted_plane.get("cut") if explicit_cut is None else explicit_cut
    cut_error = abs(cut - reference_plane["cut"]) if cut is not None else truth.shape[1]
    return dict(foreground_dice=dice(fg, gt), ap_dice=float(np.mean([dice(prediction == c, truth == c) for c in (1, 2)])),
        assd_mm=assd, hd95_mm=hd95, relative_volume_error=float((fg.sum() - gt.sum()) / gt.sum()),
        cut_error_mm=float(cut_error), cut=cut, reference_cut=reference_plane["cut"],
        ap_swaps=int(((prediction > 0) & (truth > 0) & (prediction != truth)).sum()),
        corrected_voxels=int(((raw != truth) & (prediction == truth)).sum()),
        introduced_voxels=int(((raw == truth) & (prediction != truth)).sum()),
        fg_corrected_voxels=int(((raw_fg != gt) & (fg == gt)).sum()),
        fg_introduced_voxels=int(((raw_fg == gt) & (fg != gt)).sum()),
        changed_voxels=int((prediction != raw).sum()))


def load_case(name):
    record = json.loads((REPORT / "registrations" / f"{name}.json").read_text())
    if record["prior_sha256"] != sha(CACHE / "priors" / f"{name}.npz"):
        raise ValueError("Atlas prior hash mismatch")
    with np.load(CACHE / "network" / f"{name}.npz", allow_pickle=False) as data:
        logits, truth = data["logits"].astype(np.float64), data["truth"]
    np.testing.assert_array_equal(truth, native(name, labels=True))
    with np.load(CACHE / "priors" / f"{name}.npz", allow_pickle=False) as data:
        priors = {k: data[k].astype(np.float64) for k in data.files}
    image = normalize(native(name))
    edges, weights = lattice_edges(image)
    return logits, truth, priors, edges, weights


def baseline_rows(logits, truth, priors):
    raw = logits.argmax(0).astype(np.uint8)
    plane, cut = ap_refine(logits, priors["deformable"], 0.)
    atlas_plane, atlas_cut = ap_refine(logits, priors["deformable"], 0., atlas_only=True)
    return {
        "raw": measure(raw, truth, raw),
        "model_plane": measure(plane, truth, raw, cut),
        "atlas_plane": measure(atlas_plane, truth, raw, atlas_cut),
        **{f"{kind}_atlas_alone": measure(prior.argmax(0), truth, raw) for kind, prior in priors.items()},
    }


def calibrate(cohort):
    cases = []
    for i, name in enumerate(cohort["calibration"]):
        logits, truth, priors, edges, weights = load_case(name)
        raw = logits.argmax(0).astype(np.uint8)
        row = dict(case=name, **baseline_rows(logits, truth, priors), whole={}, ap={})
        evaluated = {}
        for family, configs in WHOLE.items():
            row["whole"][family] = []
            for kind, alpha, lam in configs:
                key = (kind if alpha else "none", alpha, lam)
                if key not in evaluated:
                    pred = whole_refine(logits, priors[kind], edges, weights, alpha, lam)
                    evaluated[key] = measure(pred, truth, raw)
                    if alpha == lam == 0:
                        np.testing.assert_array_equal(pred, raw)
                row["whole"][family].append(evaluated[key])
        for family, configs in AP.items():
            row["ap"][family] = []
            for kind, alpha in configs:
                pred, cut = ap_refine(logits, priors[kind], alpha)
                row["ap"][family].append(measure(pred, truth, raw, cut))
        cases.append(row)
        print(f"Calibration scored {i+1}/{len(cohort['calibration'])}: {name}", flush=True)
    selected = dict(whole={}, ap={})
    for family, configs in WHOLE.items():
        means = [np.mean([row["whole"][family][i]["foreground_dice"] for row in cases]) for i in range(len(configs))]
        index = min(range(len(configs)), key=lambda i: (-means[i], configs[i][1], configs[i][2]))
        selected["whole"][family] = dict(config=configs[index], calibration_dice=float(means[index]))
    for family, configs in AP.items():
        means = [np.mean([row["ap"][family][i]["cut_error_mm"] for row in cases]) for i in range(len(configs))]
        index = min(range(len(configs)), key=lambda i: (means[i], configs[i][1]))
        selected["ap"][family] = dict(config=configs[index], calibration_mae=float(means[index]))
    selected["protocol_sha256"] = sha(REPORT / "PROTOCOL.md")
    selected["runner_sha256"] = sha(__file__)
    save_json(REPORT / "calibration_cases.json", cases)
    save_json(REPORT / "locked_settings.json", selected)
    print(json.dumps(selected, indent=2), flush=True)


def paired_summary(rows, arm, reference="raw"):
    a, b = [row[arm] for row in rows], [row[reference] for row in rows]
    rng = np.random.default_rng(20260924)
    indices = rng.integers(0, len(rows), size=(10000, len(rows)))
    result = {}
    for metric in ("foreground_dice", "ap_dice", "assd_mm", "hd95_mm", "cut_error_mm", "relative_volume_error"):
        values = np.array([x[metric] for x in a])
        before = np.array([x[metric] for x in b])
        delta = values - before
        direction = 1 if "dice" in metric else -1
        improvement = direction * delta if metric != "relative_volume_error" else np.abs(before) - np.abs(values)
        result[metric] = dict(mean=float(values.mean()), reference_mean=float(before.mean()), delta=float(delta.mean()),
            delta_ci95=np.quantile(delta[indices].mean(1), [.025, .975]).tolist(),
            improved=int((improvement > 1e-10).sum()), worsened=int((improvement < -1e-10).sum()),
            tied=int((np.abs(improvement) <= 1e-10).sum()))
    for key in ("ap_swaps", "corrected_voxels", "introduced_voxels", "fg_corrected_voxels", "fg_introduced_voxels", "changed_voxels"):
        result[key] = int(sum(x[key] for x in a))
    correct = [i for i, x in enumerate(b) if x["cut_error_mm"] == 0]
    result["initially_correct_cuts"] = len(correct)
    result["spoiled_correct_cuts"] = sum(a[i]["cut_error_mm"] > 0 for i in correct)
    # Oracle selection diagnoses complementary errors; unavailable at inference.
    da = np.array([x["foreground_dice"] for x in a])
    db = np.array([x["foreground_dice"] for x in b])
    ca = np.array([x["cut_error_mm"] for x in a])
    cb = np.array([x["cut_error_mm"] for x in b])
    result["oracle_case_selection"] = dict(foreground_dice_gain=float(np.maximum(da, db).mean() - db.mean()),
        cut_mae_gain=float(cb.mean() - np.minimum(ca, cb).mean()))
    return result


def evaluate(cohort, subset):
    selected = json.loads((REPORT / "locked_settings.json").read_text())
    if selected["protocol_sha256"] != sha(REPORT / "PROTOCOL.md") or selected["runner_sha256"] != sha(__file__):
        raise ValueError("Locked settings provenance mismatch")
    if subset == "development":
        decision = json.loads((REPORT / "assessment_summary.json").read_text())["advancement"]
        if not any(decision.values()):
            raise ValueError("Development evaluation not warranted by the prespecified gate")
    rows = []
    for i, name in enumerate(cohort[subset]):
        logits, truth, priors, edges, weights = load_case(name)
        raw = logits.argmax(0).astype(np.uint8)
        row = dict(case=name, **baseline_rows(logits, truth, priors))
        predictions = {"raw": raw}
        for family, chosen in selected["whole"].items():
            kind, alpha, lam = chosen["config"]
            pred = whole_refine(logits, priors[kind], edges, weights, alpha, lam)
            row[family] = measure(pred, truth, raw)
            predictions[family] = pred
        for family, chosen in selected["ap"].items():
            kind, alpha = chosen["config"]
            pred, cut = ap_refine(logits, priors[kind], alpha)
            np.testing.assert_array_equal(pred > 0, raw > 0)
            row[family] = measure(pred, truth, raw, cut)
            predictions[family] = pred
        rows.append(row)
        folder = CACHE / "predictions" / subset
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder / f"{name}.npz", truth=truth, **predictions)
        print(f"{subset.title()} scored {i+1}/{len(cohort[subset])}: {name}", flush=True)
    summary = dict(cases=len(rows), comparisons={}, advancement={})
    for arm in rows[0]:
        if arm != "case":
            summary["comparisons"][arm] = paired_summary(rows, arm)
    summary["ap_vs_model_plane"] = {arm: paired_summary(rows, arm, "model_plane") for arm in AP}
    summary["whole_vs_graph_only"] = {arm: paired_summary(rows, arm, "graph_only") for arm in ("deformable_fusion", "deformable_graph")}
    for arm in ("deformable_fusion", "deformable_graph"):
        delta = summary["comparisons"][arm]
        graph = summary["whole_vs_graph_only"][arm]
        d = delta["foreground_dice"]
        summary["advancement"][arm] = bool(d["delta"] >= .001 and d["delta_ci95"][0] > 0
            and graph["foreground_dice"]["delta"] >= .001 and delta["assd_mm"]["delta"] <= 0
            and d["improved"] >= d["worsened"])
    ap = summary["ap_vs_model_plane"]["deformable_ap"]
    summary["advancement"]["deformable_ap"] = bool(ap["cut_error_mm"]["delta"] <= -.1
        and ap["cut_error_mm"]["delta_ci95"][1] < 0 and ap["ap_dice"]["delta"] >= 0
        and summary["comparisons"]["deformable_ap"]["spoiled_correct_cuts"] <= 1)
    save_json(REPORT / f"{subset}_cases.json", rows)
    save_json(REPORT / f"{subset}_summary.json", summary)
    print(json.dumps(summary["advancement"], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("calibrate", "assessment", "development"), required=True)
    args = parser.parse_args()
    cohort = json.loads((REPORT / "cohort.json").read_text())
    if args.stage == "calibrate":
        calibrate(cohort)
    else:
        evaluate(cohort, args.stage)


if __name__ == "__main__":
    main()
