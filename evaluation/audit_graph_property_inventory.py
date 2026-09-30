"""Check the proposed anatomy-property inventory on native and cached masks.

Spectra are combinatorial volume-graph spectra, not surface ShapeDNA.
Coordinates/centerlines are explicitly geometric proxies without anatomical
landmarks. Per-component eccentricities do not hide disconnected graphs.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

import networkx as nx
import nibabel as nib
import numpy as np
import scipy
from scipy import ndimage as ndi
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh, spsolve
from scipy.spatial.distance import cdist
from scipy.stats import rankdata, spearmanr
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.voxel_graph_anatomy import voxel_graph, adjacency, digest, save_json, REGIONS
from evaluation.audit_ap_partition_topology import score


def describe(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    return dict(mean=float(x.mean()), q05=float(np.quantile(x, .05)), median=float(np.median(x)),
                q95=float(np.quantile(x, .95)), maximum=float(x.max())) if x.size else None


def spectral_fields(matrix, points, modes=12):
    """Low-frequency fields on a connected graph; check residual and basis."""
    n = matrix.shape[0]
    lap = csgraph.laplacian(matrix).astype(float).tocsr()
    if n <= modes:
        values, vectors = np.linalg.eigh(lap.toarray())
    else:
        values, vectors = eigsh(lap, k=modes, sigma=-1e-6, which="LM",
                              v0=np.random.default_rng(23).normal(size=n), tol=1e-8)
    order = np.argsort(values)
    values, vectors = np.maximum(values[order], 0), vectors[:, order]
    for k in range(vectors.shape[1]):
        if vectors[np.argmax(np.abs(vectors[:, k])), k] < 0:
            vectors[:, k] *= -1
    if n > 1 and np.dot(vectors[:, 1], points[:, 1] - points[:, 1].mean()) < 0:
        vectors[:, 1] *= -1
    residual = float(np.abs(lap @ vectors - vectors * values).max())
    orthogonality = float(np.abs(vectors.T @ vectors - np.eye(len(values))).max())
    if residual > 1e-5 or orthogonality > 1e-5:
        raise ValueError("Unreliable spectral solve")
    times = np.array([.1, .3, 1., 3.]) / max(values[1] if n > 1 else 1., 1e-12)
    hks = vectors**2 @ np.exp(-values[:, None] * times)
    tail_bound = (np.maximum(0., 1. - (vectors**2).sum(1))[:, None]
                  * np.exp(-values[-1] * times)) if len(values) < n else np.zeros_like(hks)
    return values, vectors, hks, times, dict(residual_max=residual,
        orthogonality_error=orthogonality, hks_absolute_tail_bound_max=tail_bound.max(0).tolist())


def digital_local_thickness(points, radii):
    """Largest covering digital-ball DIAMETER; EDT uses background centers.

    All foreground-centered open balls with radius EDT(center) are considered.
    This is a lattice estimate, not exact distance to the union's cube surface.
    """
    result = np.zeros(len(points))
    for start in range(0, len(points), 128):
        d = cdist(points[start:start + 128], points)
        covered = d < radii[None, :] - 1e-10
        result[start:start + 128] = np.max(np.where(covered, 2 * radii[None, :], 0), axis=1)
    return result


def cross_sections(mask):
    """Coronal slabs and their 4-connected-component quotient graph.

    This slice-component graph is a discrete Reeb-like proxy, not a computed
    continuous Reeb graph or a topology-preserving skeleton.
    """
    ids = np.full(mask.shape, -1, dtype=np.int32)
    rows, node_rows = [], []
    offset = 0
    for y in np.flatnonzero(mask.any(axis=(0, 2))):
        slab = mask[:, y, :]
        labels, count = ndi.label(slab, ndi.generate_binary_structure(2, 1))
        ids[:, y, :] = np.where(labels > 0, labels + offset - 1, -1)
        perimeter = sum(np.abs(np.diff(np.pad(slab.astype(int), 1), axis=axis)).sum() for axis in (0, 1))
        pts = np.argwhere(slab)
        rows.append(dict(y=int(y), area=int(slab.sum()), components_4=int(count),
            perimeter=int(perimeter), perimeter_over_area=float(perimeter / slab.sum()),
            lr_extent=int(np.ptp(pts[:, 0]) + 1), si_extent=int(np.ptp(pts[:, 1]) + 1)))
        for k in range(1, count + 1):
            node_rows.append(dict(y=int(y), voxels=int((labels == k).sum())))
        offset += count
    valid = (ids[:, :-1] >= 0) & (ids[:, 1:] >= 0)
    pairs = np.unique(np.column_stack((ids[:, :-1][valid], ids[:, 1:][valid])), axis=0)
    g = nx.Graph()
    g.add_nodes_from(range(offset))
    g.add_edges_from(pairs.tolist())
    branching = [i for i, d in g.degree() if d > 2]
    return rows, dict(nodes=offset, edges=len(pairs), components=nx.number_connected_components(g),
        branch_nodes=len(branching), branch_y=[node_rows[i]["y"] for i in branching],
        leaves=sum(d == 1 for _, d in g.degree()), cycle_rank=len(pairs) - offset + nx.number_connected_components(g))


def region_properties(mask, output, whole=False):
    points, edges, axes = voxel_graph(mask)
    n = len(points)
    if not n:
        return {"empty": True}
    matrix = adjacency(n, edges)
    count, component = csgraph.connected_components(matrix, directed=False)
    largest = np.flatnonzero(component == np.bincount(component).argmax())
    degree = np.diff(matrix.indptr)
    # All-pairs shortest paths give exact per-component eccentricity/diameter.
    dist = csgraph.shortest_path(matrix, directed=False, unweighted=True)
    finite = np.isfinite(dist)
    ecc = np.where(finite, dist, -1).max(1)
    lcc_dist = dist[np.ix_(largest, largest)]
    start, end = np.unravel_index(lcc_dist.argmax(), lcc_dist.shape)
    start, end = int(largest[start]), int(largest[end])
    if points[start, 1] > points[end, 1]:
        start, end = end, start
    endpoint_distances = np.column_stack((dist[start], dist[end]))
    boundary = degree < 6
    depth = dist[:, boundary].min(1)
    del dist, lcc_dist, finite
    padded = np.pad(mask, 1)
    edt = ndi.distance_transform_edt(padded)[1:-1, 1:-1, 1:-1][tuple(points.T)]
    thickness = digital_local_thickness(points, edt)
    lap = csgraph.laplacian(matrix[largest][:, largest]).tocsr()
    values, vectors, hks, times, diagnostics = spectral_fields(matrix[largest][:, largest], points[largest])
    fiedler = vectors[:, 1] if len(values) > 1 else vectors[:, 0]
    fiedler01 = (fiedler - fiedler.min()) / max(np.ptp(fiedler), 1e-12)
    quantile = (rankdata(fiedler, method="average") - .5) / len(fiedler)
    # Dirichlet endpoints: low/high 10% of RAS-y extent in LCC, not semantic ends.
    y = points[largest, 1]
    low, high = y.min(), y.max()
    known_low, known_high = y <= low + .1 * (high - low), y >= high - .1 * (high - low)
    known = known_low | known_high
    harmonic = np.zeros(len(largest))
    harmonic[known_high] = 1
    if (~known).any():
        harmonic[~known] = spsolve(lap[~known][:, ~known], -lap[~known][:, known] @ harmonic[known])
    assert np.all(harmonic >= -1e-6) and np.all(harmonic <= 1 + 1e-6)
    # Centerline PROXY: one shortest path between a diameter pair; not a medial axis.
    _, predecessors = csgraph.shortest_path(matrix, directed=False, unweighted=True, indices=start, return_predecessors=True)
    path = [end]
    while path[-1] != start:
        path.append(int(predecessors[path[-1]]))
    path = np.array(path[::-1])
    distances_to_path = cdist(points, points[path])
    nearest = distances_to_path.argmin(1)
    degree6 = float((degree == 6).mean())
    surface = int((6 - degree).sum())
    centered = points - points.mean(0)
    eigval, eigvec = np.linalg.eigh(centered.T @ centered / n)
    principal = eigvec[:, -1]
    angle = float(np.degrees(np.arccos(np.clip(abs(principal[1]), 0, 1))))
    stats = dict(nodes=n, components_6=int(count), leaves=int((degree == 1).sum()),
        degree6_fraction=degree6, surface_area_voxel_faces=surface,
        isoperimetric_ratio=float(surface**3 / (36 * np.pi * n**2)),
        elongation_pca_std_ratio=float(np.sqrt(eigval[-1] / max(eigval[0], 1e-12))),
        coronal_normal_to_pca_axis_degrees=angle,
        depth_hops=describe(depth), edt_center_radius=describe(edt), digital_local_thickness=describe(thickness),
        eccentricity_within_component=describe(ecc),
        diameter_largest_component=int(endpoint_distances[end, 0]),
        diameter_max_over_components=int(ecc.max()), global_diameter_finite=bool(count == 1),
        diameter_path_mean_depth=float(depth[path].mean()),
        diameter_path_boundary_fraction=float((depth[path] == 0).mean()),
        lambda2_full_graph=float(values[1]) if count == 1 and len(values) > 1 else 0.,
        lambda2_lcc=float(values[1]) if len(values) > 1 else 0.,
        spectrum_lcc=values.tolist(), spectrum_lcc_volume_scaled=(values * len(largest)**(2/3)).tolist(),
        fiedler_y_spearman=float(spearmanr(fiedler, y).statistic) if np.ptp(y) else None,
        spectral_diagnostics=diagnostics)
    arrays = dict(coordinates_ijk=points, degree=degree, component_ids=component, depth_hops=depth,
        edt_background_center_radius=edt, digital_local_thickness_diameter=thickness,
        eccentricity_within_component=ecc, lcc_node_ids=largest, eigenvalues=values,
        eigenvectors_lcc=vectors, hks_lcc=hks, hks_times=times, fiedler01_lcc=fiedler01,
        fiedler_volume_quantile_lcc=quantile, harmonic_lcc=harmonic,
        endpoint_node_ids=np.array([start, end]), distance_to_endpoint_hops=endpoint_distances,
        diameter_path_node_ids=path, distance_to_diameter_path=distances_to_path.min(1),
        nearest_diameter_path_arc_hops=nearest)
    if whole:
        g = nx.from_scipy_sparse_array(matrix)
        bc = nx.betweenness_centrality(g, k=min(32, n), normalized=True, seed=23)
        arrays["betweenness_32_sources"] = np.array([bc[i] for i in range(n)])
        stats["betweenness_32_sources"] = describe(arrays["betweenness_32_sources"])
    stats["coronal_profile"], stats["slice_component_graph"] = cross_sections(mask)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    return stats


def run_case(job):
    source, name, labels, out, split, sha = job
    with threadpool_limits(limits=1):
        result = dict(source=source, case=name, split=split, source_sha256=sha, regions={})
        for region, classes in REGIONS.items():
            result["regions"][region] = region_properties(np.isin(labels, classes), out / source / name / f"{region}.npz", whole=region == "whole")
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/experiments/graph_property_inventory_20260923")
    parser.add_argument("--maps", type=Path, default=ROOT / "experiments/graph_property_inventory_20260923/maps")
    args = parser.parse_args()
    original = ROOT / "docs/experiments/voxel_graph_anatomy_20260923"
    old_manifest = json.loads((original / "manifest.json").read_text())
    old_cases = {c["case"]: c for c in json.loads((original / "cases.json").read_text())}
    jobs, refs = [], {}
    for name, previous in old_cases.items():
        path = Path(old_manifest["dataset"]) / "labelsTr" / f"{name}.nii.gz"
        sha = digest(path)
        if sha != old_manifest["inputs"][name]["source_sha256"]:
            raise ValueError("Native source changed")
        image = nib.load(path)
        if not np.allclose(image.header.get_zooms(), 1):
            raise ValueError("This audit's metric implementations require unit isotropic spacing")
        labels = np.asarray(image.dataobj)
        refs[name] = labels
        jobs.append(("reference", name, labels, args.maps, previous["split"], sha))
    for arm in ("baseline_seed0", "augmentation_seed0"):
        directory = ROOT / "experiments/uncal_fold_early_stopping_20260921/voxel_audit" / arm / "error_maps"
        for name, previous in old_cases.items():
            if previous["split"] != "validation":
                continue
            path = directory / f"{name}.npz"
            with np.load(path) as z:
                pred, gt = z["prediction"], z["ground_truth"]
            pads = [((b-a)//2, (b-a+1)//2) for a, b in zip(refs[name].shape, gt.shape)]
            if not np.array_equal(np.pad(refs[name], pads), gt):
                raise ValueError("Prediction reference mismatch")
            jobs.append((arm, name, pred, args.maps, "validation", digest(path)))
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = dict(script_sha256=digest(__file__), prior_manifest_sha256=digest(original / "manifest.json"),
        maps=str(args.maps), versions=dict(numpy=np.__version__, scipy=scipy.__version__, networkx=nx.__version__),
        jobs=[dict(source=j[0], case=j[1], split=j[4], sha256=j[5]) for j in jobs],
        notes="Exploratory descriptors, no training or fitting on validation; disconnected spectral fields use LCC explicitly")
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Changed provenance; use a new output folder")
    save_json(manifest_path, manifest)
    pending = [j for j in jobs if not (args.output / "case_metrics" / j[0] / f"{j[1]}.json").exists()]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, result in enumerate(executor.map(run_case, pending)):
            save_json(args.output / "case_metrics" / result["source"] / f"{result['case']}.json", result)
            if (index + 1) % 10 == 0 or index + 1 == len(pending):
                print(f"Completed {index+1}/{len(pending)} new cases (three regions each)", flush=True)
    print("Descriptor inventory complete", flush=True)


if __name__ == "__main__":
    main()
