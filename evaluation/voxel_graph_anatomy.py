"""Native six-face voxel graphs, cubical topology, and A/P interface audit.

Graph cycles and anatomical tunnels are deliberately separate quantities.
Run with --help; outputs contain no image intensities or learned predictions
unless the optional, existing prediction-cache audit is requested.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import platform
import sys

import networkx as nx
import nibabel as nib
import numpy as np
import scipy
from scipy import ndimage as ndi, sparse
from scipy.sparse.csgraph import connected_components, shortest_path

ROOT = Path(__file__).resolve().parents[1]
REGIONS = {"whole": (1, 2), "anterior": (1,), "posterior": (2,)}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def voxel_graph(mask):
    """One node per true voxel, one undirected edge per shared face (once)."""
    mask = np.asarray(mask, bool)
    if mask.ndim != 3:
        raise ValueError("Expected a 3D mask")
    coordinates = np.argwhere(mask).astype(np.int32)
    ids = np.full(mask.shape, -1, dtype=np.int32)
    ids[mask] = np.arange(len(coordinates))
    edges, axes = [], []
    for axis in range(3):
        lo, hi = [slice(None)] * 3, [slice(None)] * 3
        lo[axis], hi[axis] = slice(None, -1), slice(1, None)
        lo, hi = tuple(lo), tuple(hi)
        valid = mask[lo] & mask[hi]
        edges.append(np.column_stack((ids[lo][valid], ids[hi][valid])))
        axes.append(np.full(valid.sum(), axis, dtype=np.uint8))
    return coordinates, np.concatenate(edges), np.concatenate(axes)


def adjacency(n, edges, weights=None):
    weights = np.ones(len(edges)) if weights is None else weights
    return sparse.csr_matrix((np.r_[weights, weights],
                              (np.r_[edges[:, 0], edges[:, 1]],
                               np.r_[edges[:, 1], edges[:, 0]])), shape=(n, n))


def graph_stats(coordinates, edges, axes, spacing):
    n, m = len(coordinates), len(edges)
    matrix = adjacency(n, edges)
    count, component = connected_components(matrix, directed=False)
    sizes = np.bincount(component)
    degree = np.diff(matrix.indptr)
    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from(edges.tolist())
    blocks = list(nx.biconnected_components(g))
    articulation = list(nx.articulation_points(g))
    bridges = [tuple(b) for b in blocks if len(b) == 2]
    lcc = int(sizes.max()) if n else 0
    diameter_lower, tortuosity = None, None
    if n:
        nodes = np.flatnonzero(component == sizes.argmax())
        weighted = adjacency(n, edges, np.asarray(spacing)[axes])
        d = shortest_path(weighted, directed=False, indices=int(nodes[0]))
        start = int(nodes[np.argmax(d[nodes])])
        d = shortest_path(weighted, directed=False, indices=start)
        end = int(nodes[np.argmax(d[nodes])])
        diameter_lower = float(d[end])
        euclidean = np.linalg.norm((coordinates[start] - coordinates[end]) * spacing)
        tortuosity = float(d[end] / euclidean) if euclidean else 1.
    return {
        "nodes": n, "edges": m, "components_6": int(count),
        "component_sizes": sorted(sizes.tolist(), reverse=True),
        "island_voxels": n - lcc,
        "largest_component_fraction": lcc / n if n else 0.,
        "cycle_rank": int(m - n + count),
        "degree_histogram": np.bincount(degree, minlength=7).tolist(),
        "mean_degree": float(degree.mean()) if n else 0.,
        "bridges": len(bridges), "articulation_voxels": len(articulation),
        "largest_biconnected_block_fraction": max(map(len, blocks), default=int(n > 0)) / n if n else 0.,
        "diameter_lower_bound_mm": diameter_lower,
        "double_sweep_path_to_chord_ratio": tortuosity,
    }, component


def cubical_stats(mask):
    """Homology of the union of CLOSED voxel cubes, including corner contacts.

    Count all cells for Euler characteristic; beta0 uses 26-connectivity and
    beta2 counts bounded 6-connected background components. Alexander duality
    plus Euler gives beta1. These are NOT Betti numbers of the six-face graph.
    """
    mask = np.asarray(mask, bool)
    if not mask.any():
        return dict(beta0=0, beta1=0, beta2=0, euler=0, cells=[0, 0, 0, 0])
    pts = np.argwhere(mask)
    cropped = mask[tuple(slice(a, b + 1) for a, b in zip(pts.min(0), pts.max(0)))]
    cells = np.zeros(tuple(2 * np.array(cropped.shape) + 1), dtype=bool)
    cells[1::2, 1::2, 1::2] = cropped
    cells = ndi.maximum_filter(cells, size=3, mode="constant", cval=0)
    counts = [0, 0, 0, 0]
    for parity in itertools.product((0, 1), repeat=3):
        counts[sum(parity)] += int(cells[tuple(slice(p, None, 2) for p in parity)].sum())
    chi = counts[0] - counts[1] + counts[2] - counts[3]
    beta0 = ndi.label(cropped, np.ones((3, 3, 3)))[1]
    background = ~np.pad(cropped, 1)
    beta2 = ndi.label(background, ndi.generate_binary_structure(3, 1))[1] - 1
    beta1 = beta0 + beta2 - chi
    if beta1 < 0:
        raise AssertionError("Invalid cubical Euler/duality computation")
    return dict(beta0=int(beta0), beta1=int(beta1), beta2=int(beta2), euler=int(chi), cells=counts)


def gf2_rank(columns):
    """Exact sparse GF(2) elimination using Python integer bit columns."""
    pivots = {}
    for col in columns:
        while col:
            pivot = col.bit_length() - 1
            if pivot not in pivots:
                pivots[pivot] = col
                break
            col ^= pivots[pivot]
    return len(pivots)


def square_corners(origin, normal):
    basis = np.eye(3, dtype=int)
    u, v = [a for a in range(3) if a != normal]
    return [tuple(origin), tuple(origin + basis[u]),
            tuple(origin + basis[u] + basis[v]), tuple(origin + basis[v])]


def square_complex_stats(squares):
    """Exact interface homology and edge/vertex manifold tests on shared faces."""
    vertices, edge_faces, face_edges, links = set(), defaultdict(list), [], defaultdict(list)
    for face_id, corners in enumerate(squares):
        vertices.update(corners)
        boundary = []
        for i, vertex in enumerate(corners):
            edge = tuple(sorted((vertex, corners[(i + 1) % 4])))
            edge_faces[edge].append(face_id)
            boundary.append(edge)
            links[vertex].append((corners[(i - 1) % 4], corners[(i + 1) % 4]))
        face_edges.append(boundary)
    g = nx.Graph()
    g.add_nodes_from(vertices)
    g.add_edges_from(edge_faces)
    b0 = nx.number_connected_components(g)
    edge_index = {edge: i for i, edge in enumerate(edge_faces)}
    rank2 = gf2_rank(sum(1 << edge_index[e] for e in face) for face in face_edges)
    v, e, f = len(vertices), len(edge_faces), len(squares)
    b1, b2 = e - v + b0 - rank2, f - rank2
    boundary_edges = [edge for edge, faces in edge_faces.items() if len(faces) == 1]
    boundary_graph = nx.Graph(boundary_edges)
    bad_edges = sum(len(faces) > 2 for faces in edge_faces.values())
    bad_vertices = 0
    for vertex, link_edges in links.items():
        link = nx.Graph(link_edges)
        degrees = [d for _, d in link.degree()]
        is_cycle = all(d == 2 for d in degrees)
        is_path = degrees.count(1) == 2 and all(d in (1, 2) for d in degrees)
        if not nx.is_connected(link) or not (is_cycle or is_path):
            bad_vertices += 1
    boundary_components = nx.number_connected_components(boundary_graph)
    boundary_is_loops = bool(boundary_edges) and all(d == 2 for _, d in boundary_graph.degree())
    is_manifold = bool(f) and bad_edges == 0 and bad_vertices == 0
    return {
        "vertices": v, "edges": e, "faces": f, "euler": v - e + f,
        "beta0": b0, "beta1": b1, "beta2": b2,
        "boundary_edges": len(boundary_edges), "boundary_components": boundary_components,
        "boundary_is_loops": boundary_is_loops,
        "nonmanifold_edges": bad_edges, "nonmanifold_vertices": bad_vertices,
        "is_manifold_with_boundary": is_manifold,
        "is_disk": is_manifold and b0 == 1 and b1 == 0 and b2 == 0
                   and boundary_components == 1 and boundary_is_loops,
    }


def partition_stats(labels, coordinates, edges, axes, spacing):
    values = labels[tuple(coordinates.T)]
    cross = values[edges[:, 0]] != values[edges[:, 1]]
    squares = []
    area = np.prod(spacing) / np.asarray(spacing)
    for (lo, _), axis in zip(edges[cross], axes[cross]):
        origin = coordinates[lo].copy()
        origin[axis] += 1
        squares.append(square_corners(origin, int(axis)))
    result = square_complex_stats(squares)
    degree = np.bincount(edges.ravel(), minlength=len(values))
    volumes = [int(degree[values == cls].sum()) for cls in (1, 2)]
    ncut = float(cross.sum() * sum(1 / v for v in volumes)) if min(volumes) > 0 else None
    result.update(cut_edges=int(cross.sum()), area_mm2=float(area[axes[cross]].sum()),
                  axis_face_counts=np.bincount(axes[cross], minlength=3).tolist(),
                  anterior_fraction=float(np.mean(values == 1)) if len(values) else None,
                  normalized_cut=ncut)
    # All coronal cuts leaving at least 10% of foreground on each side.
    # A fixed geometric diagnostic, not a hyperparameter fitted on validation.
    profile = []
    for cut in range(1, labels.shape[1]):
        right = coordinates[:, 1] >= cut
        frac = right.mean() if len(right) else 0.
        if not .1 <= frac <= .9:
            continue
        cut_edges = int(np.count_nonzero(right[edges[:, 0]] != right[edges[:, 1]]))
        vol = [int(degree[right].sum()), int(degree[~right].sum())]
        if min(vol) == 0:
            continue
        cost = cut_edges * sum(1 / v for v in vol)
        swaps = int(np.count_nonzero(np.where(right, 1, 2) != values))
        profile.append(dict(cut=cut, normalized_cut=cost, disagreement=swaps,
                            anterior_fraction=float(frac), cut_edges=cut_edges))
    # The reference best plane is allowed any nonempty side, independent of balance.
    fits = []
    for cut in range(int(coordinates[:, 1].min()) + 1, int(coordinates[:, 1].max()) + 1) if len(values) else []:
        fit = np.where(coordinates[:, 1] >= cut, 1, 2)
        fits.append((int(np.count_nonzero(fit != values)), cut))
    if fits:
        disagreement, best_cut = min(fits)
        result.update(best_plane_cut=best_cut, plane_disagreement_voxels=disagreement)
    else:
        result.update(best_plane_cut=None, plane_disagreement_voxels=None)
    if profile:
        best = min(profile, key=lambda row: (row["normalized_cut"], row["cut"]))
        result.update(min_ncut_plane=best["cut"],
                      min_ncut_plane_error_mm=abs(best["cut"] - result["best_plane_cut"]) * float(spacing[1]),
                      min_ncut_value=best["normalized_cut"])
    else:
        result.update(min_ncut_plane=None, min_ncut_plane_error_mm=None, min_ncut_value=None)
    return result, profile


def analyze(labels, spacing, affine=None, output=None, provenance=None):
    if labels.ndim != 3 or not np.isin(labels, (0, 1, 2)).all():
        raise ValueError("Expected 3D background=0, anterior=1, posterior=2 labels")
    result = {"regions": {}}
    for region, classes in REGIONS.items():
        mask = np.isin(labels, classes)
        coordinates, edges, axes = voxel_graph(mask)
        stats, components = graph_stats(coordinates, edges, axes, spacing)
        stats["cubical"] = cubical_stats(mask)
        stats["volume_mm3"] = float(mask.sum() * np.prod(spacing))
        padded = np.pad(mask, 1)
        distance = ndi.distance_transform_edt(padded, sampling=spacing)[1:-1, 1:-1, 1:-1]
        stats["erosion"] = {}
        for radius in (1., 2.):
            core = distance > radius
            ids, count = ndi.label(core, ndi.generate_binary_structure(3, 1))
            sizes = np.bincount(ids.ravel())[1:]
            stats["erosion"][str(radius)] = dict(components_6=int(count), voxels=int(core.sum()),
                largest_fraction=float(sizes.max() / sizes.sum()) if sizes.size else 0.)
        result["regions"][region] = stats
        if output is not None:
            output.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(output / f"{region}.npz", coordinates_ijk=coordinates,
                edges=edges.astype(np.int32), edge_axis=axes,
                edge_length_mm=np.asarray(spacing)[axes],
                shared_face_area_mm2=(np.prod(spacing) / np.asarray(spacing))[axes],
                node_labels=labels[tuple(coordinates.T)], component_ids=components,
                affine=affine, shape=np.array(labels.shape), spacing=np.asarray(spacing),
                metadata=json.dumps(provenance or {}))
        if region == "whole":
            result["interface"], profile = partition_stats(labels, coordinates, edges, axes, spacing)
            result["coronal_cut_profile"] = profile
    return result


def distribution(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return dict(zip(("min", "q05", "median", "q95", "max"),
                    map(float, np.quantile(values, [0, .05, .5, .95, 1]))))


def summarize(cases):
    out = {"cases": len(cases), "regions": {}}
    for region in REGIONS:
        rows = [c["regions"][region] for c in cases]
        out["regions"][region] = {
            "disconnected_cases": sum(r["components_6"] > 1 for r in rows),
            "island_voxels_total": sum(r["island_voxels"] for r in rows),
            "cases_with_bridges": sum(r["bridges"] > 0 for r in rows),
            "cases_with_articulations": sum(r["articulation_voxels"] > 0 for r in rows),
            "cubical_betti_counts": dict(Counter(str([r["cubical"][f"beta{i}"] for i in range(3)]) for r in rows)),
            **{key: distribution([r[key] for r in rows]) for key in (
                "nodes", "edges", "cycle_rank", "island_voxels", "bridges", "articulation_voxels",
                "largest_biconnected_block_fraction", "diameter_lower_bound_mm", "double_sweep_path_to_chord_ratio")}}
    rows = [c["interface"] for c in cases]
    out["interface"] = {
        "disk_cases": sum(r["is_disk"] for r in rows),
        "manifold_cases": sum(r["is_manifold_with_boundary"] for r in rows),
        "betti_counts": dict(Counter(str([r[f"beta{i}"] for i in range(3)]) for r in rows)),
        "nonplanar_cases": sum((r["plane_disagreement_voxels"] or 0) > 0 for r in rows),
        "plane_disagreement_voxels_total": sum(r["plane_disagreement_voxels"] or 0 for r in rows),
        "min_ncut_matches_best_plane": sum(r["min_ncut_plane_error_mm"] == 0 for r in rows),
        **{key: distribution([r[key] for r in rows]) for key in (
            "cut_edges", "normalized_cut", "anterior_fraction", "min_ncut_plane_error_mm")}}
    return out


def compare_predictions(args, split, native, rows, manifest):
    output, details = {}, {}
    for arm in ("baseline_seed0", "augmentation_seed0"):
        directory = args.predictions / arm / "error_maps"
        if {p.stem for p in directory.glob("*.npz")} != set(split["val"]):
            raise ValueError(f"Prediction membership mismatch: {directory}")
        arm_rows = []
        for name in split["val"]:
            path = directory / f"{name}.npz"
            with np.load(path, allow_pickle=False) as archive:
                gt, pred = archive["ground_truth"], archive["prediction"]
            shape = native[name].shape
            if pred.shape != gt.shape or any(a > b for a, b in zip(shape, gt.shape)):
                raise ValueError(f"Prediction geometry mismatch: {name}")
            pads = [((b - a) // 2, (b - a + 1) // 2) for a, b in zip(shape, gt.shape)]
            if not np.array_equal(np.pad(native[name], pads), gt):
                raise ValueError(f"Prediction reference mismatch: {name}")
            manifest["prediction_hashes"][str(path)] = digest(path)
            analyzed = analyze(pred, rows[name]["spacing"])
            swaps = int(((pred > 0) & (gt > 0) & (pred != gt)).sum())
            reference = rows[name]
            # Case-specific GT matching is a diagnostic, never an inference feature.
            betti_match = all(
                all(analyzed["regions"][r]["cubical"][f"beta{i}"] == reference["regions"][r]["cubical"][f"beta{i}"] for i in range(3))
                for r in REGIONS)
            face_match = all(analyzed["regions"][r]["components_6"] == reference["regions"][r]["components_6"] for r in REGIONS)
            interface_match = all(analyzed["interface"][f"beta{i}"] == reference["interface"][f"beta{i}"] for i in range(3))
            arm_rows.append(dict(case=name, ap_swaps=swaps, cubical_betti_match=betti_match,
                graph_components_match=face_match, interface_betti_match=interface_match,
                **analyzed))
        summary = summarize(arm_rows)
        summary["ap_swaps_total"] = sum(r["ap_swaps"] for r in arm_rows)
        summary["reference_matching"] = {}
        for key in ("cubical_betti_match", "graph_components_match", "interface_betti_match"):
            matched = [r for r in arm_rows if r[key]]
            summary["reference_matching"][key] = dict(cases=len(matched), ap_swaps=sum(r["ap_swaps"] for r in matched))
        matched = [r for r in arm_rows if all(r[k] for k in ("cubical_betti_match", "graph_components_match", "interface_betti_match"))]
        summary["reference_matching"]["all_three"] = dict(cases=len(matched), ap_swaps=sum(r["ap_swaps"] for r in matched))
        output[arm], details[arm] = summary, arm_rows
        print(f"Prediction audit: {arm}, {len(arm_rows)} cases", flush=True)
    save_json(args.output / "prediction_cases.json", details)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets/Dataset101_MSD")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/experiments/voxel_graph_anatomy_20260923")
    parser.add_argument("--graphs", type=Path, default=ROOT / "experiments/voxel_graph_anatomy_20260923/graphs")
    parser.add_argument("--predictions", type=Path, help="Optional root of paired existing prediction caches")
    args = parser.parse_args()
    split_path = args.dataset / "splits_final.json"
    all_splits = json.loads(split_path.read_text())
    split = all_splits[args.fold]
    names = split["train"] + split["val"]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate or overlapping cases")
    if set(names) != {p.name.removesuffix(".nii.gz") for p in (args.dataset / "labelsTr").glob("*.nii.gz")}:
        raise ValueError("Split does not cover labelsTr exactly")
    manifest = dict(script_sha256=digest(__file__), split_sha256=digest(split_path),
        dataset_json_sha256=digest(args.dataset / "dataset.json"), dataset=str(args.dataset.resolve()),
        fold=args.fold, graph_root=str(args.graphs.resolve()), inputs={}, prediction_hashes={},
        versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                      nibabel=nib.__version__, networkx=nx.__version__),
        graph_definition="Undirected unweighted 6-face adjacency; one edge per shared foreground face",
        cubical_definition="Closed voxel union; 26 foreground / 6 background; distinct from voxel graph")
    rows, native = {}, {}
    for index, name in enumerate(names):
        path = args.dataset / "labelsTr" / f"{name}.nii.gz"
        image = nib.load(path)
        if nib.aff2axcodes(image.affine) != ("R", "A", "S"):
            raise ValueError(f"Expected RAS for AP plane diagnostic: {path}")
        gram = image.affine[:3, :3].T @ image.affine[:3, :3]
        if not np.allclose(gram, np.diag(np.diag(gram)), atol=1e-6):
            raise ValueError(f"Sheared voxels are unsupported: {path}")
        spacing = np.sqrt(np.diag(gram))
        labels = np.asarray(image.dataobj)
        native[name] = labels
        meta = dict(case=name, source_sha256=digest(path),
            train_folds=[i for i, s in enumerate(all_splits) if name in s["train"]],
            validation_folds=[i for i, s in enumerate(all_splits) if name in s["val"]])
        manifest["inputs"][name] = meta
        rows[name] = dict(case=name, split="train" if name in split["train"] else "validation",
                         spacing=spacing.tolist(), **analyze(labels, spacing, image.affine, args.graphs / name, meta))
        if (index + 1) % 20 == 0 or index + 1 == len(names):
            print(f"Reference graphs: {index + 1}/{len(names)}", flush=True)
    summaries = {subset: summarize([rows[name] for name in split[key]]) for subset, key in (("train", "train"), ("validation", "val"))}
    summaries["graph_count"] = 3 * len(names)
    summaries["all_folds"] = {str(i): {subset: summarize([rows[n] for n in s[key]])
        for subset, key in (("train", "train"), ("validation", "val"))} for i, s in enumerate(all_splits)}
    if args.predictions:
        summaries["predictions"] = compare_predictions(args, split, native, rows, manifest)
    save_json(args.output / "cases.json", list(rows.values()))
    save_json(args.output / "summary.json", summaries)
    save_json(args.output / "manifest.json", manifest)
    print(f"Saved {summaries['graph_count']} graphs and summaries to {args.output}", flush=True)


if __name__ == "__main__":
    main()
