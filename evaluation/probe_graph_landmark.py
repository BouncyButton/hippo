"""Label-blind whole-graph local landmark search; see dated protocol."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import scipy
from scipy import ndimage as ndi, sparse
from scipy.sparse.csgraph import connected_components
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.voxel_graph_anatomy import voxel_graph, adjacency, digest, save_json

SEED = 20260924
ARMS = ("position", "shape", "graph", "combined", "shuffled_graph")
RULES = ("area_rise", "thickness_rise", "degree6_rise", "betweenness_peak",
         "branch_peak", "geodesic_ncut_min", "geodesic_thickness_rise", "change_magnitude")
PRIORS = ("median_position", "median_volume")
FIELDS = ("coordinates_ijk", "degree", "depth_hops", "digital_local_thickness_diameter",
          "betweenness_32_sources", "eccentricity_within_component", "lcc_node_ids",
          "harmonic_lcc", "fiedler01_lcc", "distance_to_endpoint_hops")


def quotient_profiles(edges, bins, count):
    """Components of bins and degree>2 nodes of their component quotient graph."""
    n = len(bins)
    same = (bins[edges[:, 0]] == bins[edges[:, 1]]) & (bins[edges[:, 0]] >= 0)
    _, ids = connected_components(adjacency(n, edges[same]), directed=False)
    cross = (ids[edges[:, 0]] != ids[edges[:, 1]]) & (bins[edges[:, 0]] >= 0) & (bins[edges[:, 1]] >= 0)
    pairs = np.unique(np.sort(ids[edges[cross]], axis=1), axis=0)
    degree = np.bincount(pairs.ravel(), minlength=ids.max() + 1)
    components, branches = np.zeros(count), np.zeros(count)
    for b in range(count):
        selected = bins == b
        if selected.any():
            components[b] = len(np.unique(ids[selected]))
            branches[b] = np.mean(degree[ids[selected]] > 2)
    return components, branches


def pool(values, bins, count):
    output = np.zeros(count)
    valid = np.isfinite(values) & (bins >= 0)
    for b in range(count):
        selected = valid & (bins == b)
        if selected.any():
            output[b] = np.mean(values[selected])
    return output


def local_columns(profile, locations):
    smooth = ndi.gaussian_filter1d(np.asarray(profile, float), 1., mode="nearest")
    grid = np.arange(len(smooth))
    value = np.interp(locations, grid, smooth)
    derivative = np.interp(locations, grid, np.gradient(smooth))
    change = np.interp(locations + 2, grid, smooth) - np.interp(locations - 2, grid, smooth)
    return np.column_stack((value, derivative, change))


def extract_features(whole, shape):
    """Only the whitelisted WHOLE-graph arrays are accepted; no A/P labels."""
    points = np.asarray(whole["coordinates_ijk"])
    mask = np.zeros(shape, bool)
    mask[tuple(points.T)] = True
    rebuilt, edges, _ = voxel_graph(mask)
    np.testing.assert_array_equal(points, rebuilt)
    n = len(points)
    # A support-relative profile with fixed zero margins makes smoothing
    # independent of the image crop/padding. Return native cut indices below.
    origin = int(points[:, 1].min()) - 4
    y = points[:, 1] - origin
    length = int(y.max()) + 5
    degree = np.bincount(edges.ravel(), minlength=n)
    np.testing.assert_array_equal(degree, whole["degree"])
    area = np.bincount(y, minlength=length).astype(float)
    cumulative = np.r_[0, np.cumsum(area)]
    candidates = np.arange(int(y.min()) + 1, int(y.max()) + 1)
    fractions = cumulative[candidates] / n
    candidates = candidates[(fractions >= .1) & (fractions <= .9)]
    if len(candidates) < 2:
        raise ValueError("Insufficient nondegenerate candidate planes")
    fractions = cumulative[candidates] / n
    position = (candidates - .5 - y.min()) / (y.max() - y.min())
    pos = np.column_stack([v**power for v in (position, fractions) for power in (1, 2, 3)])
    size = n ** (1 / 3)
    widths = np.zeros((length, 2))
    for b in np.flatnonzero(area):
        widths[b] = np.ptp(points[y == b][:, [0, 2]], axis=0) + 1
    within = y[edges[:, 0]] == y[edges[:, 1]]
    perimeter = 4 * area - 2 * np.bincount(y[edges[within, 0]], minlength=length)
    degree_by_y = np.bincount(y, weights=degree, minlength=length)
    volume = np.r_[0, degree_by_y.cumsum()]
    crossing = ~within
    cut_count = np.bincount(np.maximum(y[edges[crossing, 0]], y[edges[crossing, 1]]), minlength=length)
    ncut = cut_count * (1 / np.maximum(volume[:-1], 1) + 1 / np.maximum(volume[-1] - volume[:-1], 1))
    shape_profiles = dict(area=area / size**2, perimeter_area=perimeter / np.maximum(area, 1),
                          lr_width=widths[:, 0] / size, si_width=widths[:, 1] / size, ncut=ncut * size)
    components, branches = quotient_profiles(edges, y, length)
    graph_profiles = {f"degree{k}": pool((degree == k).astype(float), y, length) for k in range(7)}
    graph_profiles.update(depth=pool(whole["depth_hops"], y, length) / size,
        thickness=pool(whole["digital_local_thickness_diameter"], y, length) / size,
        betweenness=pool(whole["betweenness_32_sources"], y, length) * n,
        eccentricity=pool(whole["eccentricity_within_component"], y, length) / size,
        components=components, branch_mass=branches)
    lcc = whole["lcc_node_ids"]
    for key in ("harmonic", "fiedler01"):
        field = np.full(n, np.nan)
        field[lcc] = whole[f"{key}_lcc"]
        graph_profiles[key] = pool(field, y, length)
    distances = whole["distance_to_endpoint_hops"]
    valid = np.isfinite(distances).all(1) & (distances.sum(1) > 0)
    u = np.full(n, np.nan)
    u[valid] = distances[valid, 0] / distances[valid].sum(1)
    graph_profiles["geodesic_u"] = pool(u, y, length)
    graph_profiles["intrinsic_valid_fraction"] = pool(valid.astype(float), y, length)
    bins = np.full(n, -1, dtype=int)
    bins[valid] = np.minimum((32 * u[valid]).astype(int), 31)
    geocount = np.bincount(bins[valid], minlength=32)
    gc, gb = quotient_profiles(edges, bins, 32)
    geo_profiles = dict(occupancy=geocount / n, thickness=pool(whole["digital_local_thickness_diameter"], bins, 32) / size,
        core_fraction=pool((degree == 6).astype(float), bins, 32), components=gc, branch_mass=gb)
    geo_ncut = np.zeros(32)
    for b in range(32):
        side = u >= (b + .5) / 32
        evalid = valid[edges].all(1)
        cut = (side[edges[:, 0]] != side[edges[:, 1]]) & evalid
        vl, vr = degree[valid & ~side].sum(), degree[valid & side].sum()
        geo_ncut[b] = cut.sum() * (1 / max(vl, 1) + 1 / max(vr, 1)) * size
    geo_profiles["ncut"] = geo_ncut
    locations = candidates - .5
    shapes = np.column_stack([local_columns(p, candidates if key == "ncut" else locations)
                             for key, p in shape_profiles.items()])
    graphs = np.column_stack([local_columns(p, locations) for p in graph_profiles.values()])
    # Associate a coronal candidate with mean intrinsic coordinate of its cut endpoints.
    candidate_u = []
    for cut in candidates:
        cross = (y[edges[:, 0]] < cut) != (y[edges[:, 1]] < cut)
        nodes = np.unique(edges[cross])
        nodes = nodes[valid[nodes]]
        candidate_u.append(float(u[nodes].mean()) if len(nodes) else float(np.interp(cut - .5, np.arange(length), graph_profiles["geodesic_u"])))
    geolocations = np.array(candidate_u) * 32 - .5
    graphs = np.column_stack((graphs, *[local_columns(p, geolocations) for p in geo_profiles.values()]))
    gradients = np.column_stack([local_columns(p / max(np.std(p), 1e-8), locations)[:, 1]
        for p in list(shape_profiles.values())[:4] + [graph_profiles[k] for k in ("depth", "thickness", "betweenness", "degree6", "components", "branch_mass")]])
    rules = np.column_stack((
        local_columns(shape_profiles["area"], locations)[:, 1],
        local_columns(graph_profiles["thickness"], locations)[:, 1],
        local_columns(graph_profiles["degree6"], locations)[:, 1],
        local_columns(graph_profiles["betweenness"], locations)[:, 0],
        local_columns(graph_profiles["branch_mass"], locations)[:, 0],
        -local_columns(geo_profiles["ncut"], geolocations)[:, 0],
        local_columns(geo_profiles["thickness"], geolocations)[:, 1],
        np.linalg.norm(gradients, axis=1)))
    result = dict(candidates=candidates + origin, position=pos, shape=shapes, graph=graphs, rules=rules,
        normalized_position=position, posterior_fraction=fractions,
        graph_names=np.array([f"{k}_{s}" for k in graph_profiles for s in ("value", "derivative", "change")]
            + [f"geodesic_{k}_{s}" for k in geo_profiles for s in ("value", "derivative", "change")]),
        shape_names=np.array([f"{k}_{s}" for k in shape_profiles for s in ("value", "derivative", "change")]))
    if not all(np.isfinite(result[k]).all() for k in ("position", "shape", "graph", "rules")):
        raise ValueError("Nonfinite features")
    return result


def target_cut(labels):
    if not np.isin(labels, (0, 1, 2)).all() or not (labels == 1).any() or not (labels == 2).any():
        raise ValueError("Both semantic labels are required for scoring")
    a, p = (labels == 1).sum((0, 2)), (labels == 2).sum((0, 2))
    occupied = np.flatnonzero(a + p)
    cuts = np.arange(occupied.min() + 1, occupied.max() + 1)
    ca, cp = np.r_[0, a.cumsum()], np.r_[0, p.cumsum()]
    cost = ca[cuts] + cp[-1] - cp[cuts]
    return int(cuts[np.argmin(cost)])


def shuffled_graph(case):
    block = case["graph"]
    seed = int.from_bytes(hashlib.sha256(case["name"].encode()).digest()[:8], "little")
    offset = int(np.random.default_rng(seed).integers(1, len(block)))
    return np.roll(block, offset, axis=0)


def design(case, arm):
    blocks = [case["position"]]
    if arm in ("shape", "combined", "shuffled_graph"):
        blocks.append(case["shape"])
    if arm in ("graph", "combined"):
        blocks.append(case["graph"])
    if arm == "shuffled_graph":
        blocks.append(shuffled_graph(case))
    return np.column_stack(blocks)


def fit(cases, arm):
    xs, ys, weights = [], [], []
    for case in cases:
        target = case["candidates"] == case["target"]
        if target.sum() != 1:
            raise ValueError(f"Target outside candidate support: {case['name']}")
        xs.append(design(case, arm))
        ys.append(target)
        weights.append(np.where(target, .5, .5 / (len(target) - 1)))
    model = make_pipeline(StandardScaler(), LogisticRegression(C=.1, solver="lbfgs", max_iter=3000, tol=1e-7))
    model.fit(np.concatenate(xs), np.concatenate(ys), logisticregression__sample_weight=np.concatenate(weights))
    if model[-1].n_iter_.max() >= 3000:
        raise RuntimeError("Readout did not converge")
    return model


def predict(case, model, arm):
    scores = model.decision_function(design(case, arm))
    return int(case["candidates"][np.argmax(scores)])


def prior_values(cases):
    return {key: float(np.median([c[field][np.flatnonzero(c["candidates"] == c["target"])[0]] for c in cases]))
        for key, field in zip(PRIORS, ("normalized_position", "posterior_fraction"))}


def fixed_predictions(case, priors):
    out = {name: int(case["candidates"][np.argmax(case["rules"][:, i])]) for i, name in enumerate(RULES)}
    out.update({key: int(case["candidates"][np.argmin(abs(case[field] - priors[key]))])
        for key, field in zip(PRIORS, ("normalized_position", "posterior_fraction"))})
    return out


def endpoints(rows):
    values = np.array([r["error_mm"] for r in rows])
    result = dict(cases=len(values), mae_mm=float(values.mean()), median_mm=float(np.median(values)),
        p95_mm=float(np.quantile(values, .95)), exact_cases=int((values == 0).sum()),
        within1_cases=int((values <= 1).sum()), within2_cases=int((values <= 2).sum()))
    if "score_span" in rows[0]:
        result.update(flat_profile_cases=sum(r["score_span"] < 1e-10 for r in rows),
                      unique_peak_cases=sum(r["peak_ties"] == 1 for r in rows))
    return result


def paired(first, second):
    delta = np.array([a["error_mm"] - b["error_mm"] for a, b in zip(first, second)])
    draws = np.random.default_rng(SEED).choice(delta, (10000, len(delta)), replace=True).mean(1)
    return dict(mae_delta_mm=float(delta.mean()), ci95=np.quantile(draws, [.025, .975]).tolist(),
                improved=int((delta < 0).sum()), worsened=int((delta > 0).sum()), tied=int((delta == 0).sum()))


def row(case, cut, **extra):
    return dict(case=case["name"], target=case["target"], cut=int(cut),
        error_mm=float(abs(cut - case["target"])), **extra)


def rule_info(case, method):
    if method not in RULES:
        return {}
    scores = case["rules"][:, RULES.index(method)]
    return dict(score_span=float(np.ptp(scores)), peak_ties=int(np.isclose(scores, scores.max(), atol=1e-10, rtol=0).sum()))


def load_case(args, name, source, inventory):
    map_path = args.maps / source / name / "whole.npz"
    scalar_path = args.inventory / "case_metrics" / source / f"{name}.json"
    scalars = json.loads(scalar_path.read_text())
    if source == "reference":
        path = args.dataset / "labelsTr" / f"{name}.nii.gz"
        image = nib.load(path)
        if nib.aff2axcodes(image.affine) != ("R", "A", "S") or not np.allclose(image.header.get_zooms(), 1):
            raise ValueError("Expected 1-mm RAS images")
        labels = np.asarray(image.dataobj)
        prediction = None
    else:
        path = args.predictions / source / "error_maps" / f"{name}.npz"
        with np.load(path, allow_pickle=False) as archive:
            labels, prediction = archive["ground_truth"], archive["prediction"]
        native = np.asarray(nib.load(args.dataset / "labelsTr" / f"{name}.nii.gz").dataobj)
        if any(a > b for a, b in zip(native.shape, labels.shape)):
            raise ValueError("Cached shape mismatch")
        pads = [((b-a)//2, (b-a+1)//2) for a, b in zip(native.shape, labels.shape)]
        if not np.array_equal(np.pad(native, pads), labels):
            raise ValueError("Cached reference mismatch")
    if digest(path) != scalars["source_sha256"]:
        raise ValueError("Existing whole maps have different source provenance")
    with np.load(map_path, allow_pickle=False) as archive:
        whole = {k: archive[k] for k in FIELDS}
    support = labels > 0 if prediction is None else prediction > 0
    np.testing.assert_array_equal(whole["coordinates_ijk"], np.argwhere(support))
    features = extract_features(whole, labels.shape)
    features.update(name=name, target=target_cut(labels))
    inventory[f"{source}/{name}"] = dict(input_sha256=digest(path), whole_map_sha256=digest(map_path),
                                        scalar_sha256=digest(scalar_path))
    cache_path = args.cache / source / f"{name}.npz"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **features)
    return features, labels, prediction


def source_manifest(args):
    paths = [Path(__file__), args.output / "PROTOCOL.md", args.dataset / "splits_final.json",
             ROOT / "evaluation/voxel_graph_anatomy.py", ROOT / "evaluation/audit_graph_property_inventory.py",
             args.inventory / "manifest.json"]
    return dict(files={str(p.relative_to(ROOT)): digest(p) for p in paths},
        versions=dict(numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__, nibabel=nib.__version__), seed=SEED)


def training(args, split):
    manifest = source_manifest(args)
    inventory, cases = {}, []
    for i, name in enumerate(split["train"]):
        cases.append(load_case(args, name, "reference", inventory)[0])
        if (i + 1) % 40 == 0:
            print(f"Training extraction {i+1}/{len(split['train'])}", flush=True)
    oof = {name: [] for name in ARMS + RULES + PRIORS}
    fold_summary = []
    for fold, (train_ids, test_ids) in enumerate(KFold(4, shuffle=True, random_state=SEED).split(cases)):
        train, test = [cases[i] for i in train_ids], [cases[i] for i in test_ids]
        priors = prior_values(train)
        current = {name: [] for name in oof}
        for arm in ARMS:
            model = fit(train, arm)
            current[arm] = [row(c, predict(c, model, arm), cv_fold=fold) for c in test]
        for c in test:
            for name, cut in fixed_predictions(c, priors).items():
                current[name].append(row(c, cut, cv_fold=fold, **rule_info(c, name)))
        for name in oof:
            oof[name].extend(current[name])
        fold_summary.append({name: endpoints(values) for name, values in current.items()})
        print(f"Training CV fold {fold+1}/4 complete", flush=True)
    summary = {name: endpoints(values) for name, values in oof.items()}
    selected = min(ARMS[:-1], key=lambda a: summary[a]["mae_mm"])
    selected_rule = min(RULES, key=lambda a: summary[a]["mae_mm"])
    comparison = paired(oof["combined"], oof["shape"])
    improving_folds = sum(s["combined"]["mae_mm"] < s["shape"]["mae_mm"] for s in fold_summary)
    selection = dict(selected_arm=selected, selected_rule=selected_rule,
        graph_increment= comparison, improving_folds=improving_folds,
        graph_gate_passed=comparison["mae_delta_mm"] <= -.25 and comparison["ci95"][1] < 0 and improving_folds >= 3,
        candidate_coverage=sum(c["target"] in c["candidates"] for c in cases), cases=len(cases))
    save_json(args.output / "training_oof.json", oof)
    save_json(args.output / "training_summary.json", dict(methods=summary, folds=fold_summary,
        combined_vs_shuffled=paired(oof["combined"], oof["shuffled_graph"])))
    save_json(args.output / "selection.json", selection)
    save_json(args.output / "training_manifest.json", dict(**manifest, inputs=inventory,
        selection_sha256=digest(args.output / "selection.json")))
    print(json.dumps(selection, indent=2), flush=True)


def segmentation_score(pred, gt):
    dice = [2 * np.count_nonzero((pred == k) & (gt == k)) / max(np.count_nonzero(pred == k) + np.count_nonzero(gt == k), 1) for k in (1, 2)]
    return dict(ap_dice=float(np.mean(dice)), ap_swaps=int(((pred > 0) & (gt > 0) & (pred != gt)).sum()))


def evaluate(args, split):
    train_manifest = json.loads((args.output / "training_manifest.json").read_text())
    current = source_manifest(args)
    if any(train_manifest[k] != current[k] for k in current):
        raise ValueError("Source/protocol changed after selection")
    if digest(args.output / "selection.json") != train_manifest["selection_sha256"]:
        raise ValueError("Selection changed")
    cases = []
    for name in split["train"]:
        with np.load(args.cache / "reference" / f"{name}.npz", allow_pickle=False) as z:
            case = dict(z)
        case.update(name=str(case["name"]), target=int(case["target"]))
        cases.append(case)
    models = {arm: fit(cases, arm) for arm in ARMS}
    priors = prior_values(cases)
    inventory, result, summaries = {}, {}, {}
    selection = json.loads((args.output / "selection.json").read_text())
    for source in ("reference", "baseline_seed0", "augmentation_seed0"):
        methods = ARMS + RULES + PRIORS
        rows = {m: [] for m in methods}
        if source != "reference":
            rows.update(original=[], fitted_plane=[])
        coverage = 0
        for i, name in enumerate(split["val"]):
            case, truth, pred = load_case(args, name, source, inventory)
            coverage += int(case["target"] in case["candidates"])
            cuts = {arm: predict(case, model, arm) for arm, model in models.items()}
            cuts.update(fixed_predictions(case, priors))
            if pred is not None:
                original_cut = target_cut(pred)
                rows["original"].append(row(case, original_cut, **segmentation_score(pred, truth)))
                cuts["fitted_plane"] = original_cut
            for method, cut in cuts.items():
                extra = {}
                if pred is not None:
                    rendered = np.where(pred > 0, np.where(np.arange(pred.shape[1])[None, :, None] >= cut, 1, 2), 0)
                    extra = segmentation_score(rendered, truth)
                    extra["original_cut_correct"] = original_cut == case["target"]
                rows[method].append(row(case, cut, **rule_info(case, method), **extra))
            if (i + 1) % 26 == 0:
                print(f"Validation {source}: {i+1}/52", flush=True)
        result[source] = rows
        summaries[source] = dict(methods={m: endpoints(r) for m, r in rows.items()},
            candidate_coverage=coverage, combined_vs_shape=paired(rows["combined"], rows["shape"]),
            combined_vs_shuffled=paired(rows["combined"], rows["shuffled_graph"]))
        if source != "reference":
            summaries[source]["selected_vs_original"] = paired(rows[selection["selected_arm"]], rows["original"])
            for method in ARMS + ("fitted_plane", "original"):
                r = rows[method]
                summaries[source]["methods"][method].update(ap_dice=float(np.mean([x["ap_dice"] for x in r])),
                    ap_swaps=sum(x["ap_swaps"] for x in r),
                    originally_correct_cases=sum(x["error_mm"] == 0 for x in rows["original"]),
                    originally_correct_moved=sum(a["error_mm"] == 0 and b["cut"] != a["cut"] for a, b in zip(rows["original"], r)))
    save_json(args.output / "validation_cases.json", result)
    save_json(args.output / "validation_summary.json", summaries)
    save_json(args.output / "validation_manifest.json", dict(**current, inputs=inventory,
        training_manifest_sha256=digest(args.output / "training_manifest.json")))
    print(json.dumps({s: v["methods"][selection["selected_arm"]] for s, v in summaries.items()}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("train", "evaluate"), required=True)
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets/Dataset101_MSD")
    parser.add_argument("--maps", type=Path, default=ROOT / "experiments/graph_property_inventory_20260923/maps")
    parser.add_argument("--inventory", type=Path, default=ROOT / "docs/experiments/graph_property_inventory_20260923")
    parser.add_argument("--predictions", type=Path, default=ROOT / "experiments/uncal_fold_early_stopping_20260921/voxel_audit")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/experiments/graph_landmark_20260924")
    parser.add_argument("--cache", type=Path, default=ROOT / "experiments/graph_landmark_20260924/features")
    args = parser.parse_args()
    split = json.loads((args.dataset / "splits_final.json").read_text())[0]
    if set(split["train"]) & set(split["val"]) or len(set(split["train"] + split["val"])) != 260:
        raise ValueError("Invalid case split")
    with threadpool_limits(limits=1):
        (training if args.stage == "train" else evaluate)(args, split)


if __name__ == "__main__":
    main()
