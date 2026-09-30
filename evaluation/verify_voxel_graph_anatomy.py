"""Round-trip all saved graphs and measure one-voxel topology sensitivity."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

from voxel_graph_anatomy import ROOT, REGIONS, cubical_stats, digest, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "docs/experiments/voxel_graph_anatomy_20260923")
    args = parser.parse_args()
    manifest = json.loads((args.report / "manifest.json").read_text())
    cases = json.loads((args.report / "cases.json").read_text())
    directory = Path(manifest["graph_root"])
    dataset = Path(manifest["dataset"])
    source_path = ROOT / "evaluation/voxel_graph_anatomy.py"
    assert digest(source_path) == manifest["script_sha256"]
    hashes, sensitivity, shifted = {}, [], []
    for case in cases:
        name = case["case"]
        source = dataset / "labelsTr" / f"{name}.nii.gz"
        assert digest(source) == manifest["inputs"][name]["source_sha256"]
        image = nib.load(source)
        labels = np.asarray(image.dataobj)
        entry = dict(case=name, split=case["split"], regions={})
        region_edges = {}
        for region, classes in REGIONS.items():
            path = directory / name / f"{region}.npz"
            with np.load(path, allow_pickle=False) as archive:
                coords, edges, axes = [archive[k] for k in ("coordinates_ijk", "edges", "edge_axis")]
                mask = np.isin(labels, classes)
                assert np.array_equal(coords, np.argwhere(mask))
                assert np.array_equal(archive["node_labels"], labels[tuple(coords.T)])
                assert np.array_equal(archive["affine"], image.affine)
                assert json.loads(str(archive["metadata"])) == manifest["inputs"][name]
                # Independently enumerate all node-coordinate +axis neighbors.
                lookup = {tuple(p): i for i, p in enumerate(coords)}
                expected = set()
                for i, p in enumerate(coords):
                    for axis in range(3):
                        neighbor = p.copy()
                        neighbor[axis] += 1
                        j = lookup.get(tuple(neighbor))
                        if j is not None:
                            expected.add((i, j, axis))
                observed = {(int(i), int(j), int(a)) for (i, j), a in zip(edges, axes)}
                assert observed == expected and len(observed) == len(edges)
                assert len(coords) == case["regions"][region]["nodes"]
                assert len(edges) == case["regions"][region]["edges"]
                degree = np.bincount(edges.ravel(), minlength=len(coords))
                assert np.array_equal(np.bincount(degree, minlength=7), case["regions"][region]["degree_histogram"])
                if np.allclose(case["spacing"], 1):
                    assert (degree == 6).sum() == case["regions"][region]["erosion"]["1.0"]["voxels"]
                if region == "whole":
                    # On this lattice, a degree-six node retains all neighbors
                    # after a plane cut iff its y coordinate is >=cut+1 (A)
                    # or <cut-1 (P). No GT except the starting plane is used.
                    for offset in (-6, -3, -1, 0, 1, 3, 6):
                        cut = case["interface"]["best_plane_cut"] + offset
                        a = coords[:, 1] >= cut
                        if not a.any() or a.all():
                            continue
                        core_a = ((degree == 6) & (coords[:, 1] >= cut + 1)).sum() / a.sum()
                        core_p = ((degree == 6) & (coords[:, 1] < cut - 1)).sum() / (~a).sum()
                        shifted.append(dict(case=name, split=case["split"], offset_slices=offset,
                            anterior_fraction=float(a.mean()),
                            anterior_core_fraction=float(core_a), posterior_core_fraction=float(core_p),
                            anterior_more_compact=bool(core_a > core_p)))
                component_count = ndi.label(mask, ndi.generate_binary_structure(3, 1))[1]
                assert component_count == case["regions"][region]["components_6"]
                region_edges[region] = len(edges)
            hashes[str(path.relative_to(ROOT))] = digest(path)
            # Explicit background component sizes, with outside excluded.
            bg_ids, count = ndi.label(~np.pad(mask, 1), ndi.generate_binary_structure(3, 1))
            outside = int(bg_ids[0, 0, 0])
            sizes = np.bincount(bg_ids.ravel())
            cavity_sizes = [int(sizes[i]) for i in range(1, count + 1) if i != outside]
            assert len(cavity_sizes) == case["regions"][region]["cubical"]["beta2"]
            # This is binary dilation sensitivity, NOT persistence or label repair.
            dilated = ndi.binary_dilation(np.pad(mask, 2), structure=ndi.generate_binary_structure(3, 1))
            entry["regions"][region] = dict(cavity_sizes_voxels=cavity_sizes,
                                           one_face_dilation=cubical_stats(dilated))
        assert region_edges["whole"] == region_edges["anterior"] + region_edges["posterior"] + case["interface"]["cut_edges"]
        sensitivity.append(entry)
    summary = {}
    for subset in ("train", "validation"):
        rows = [c for c in sensitivity if c["split"] == subset]
        summary[subset] = {}
        for region in REGIONS:
            values = [c["regions"][region] for c in rows]
            cavity_sizes = [n for v in values for n in v["cavity_sizes_voxels"]]
            summary[subset][region] = dict(
                cavity_sizes_voxels=dict(Counter(map(str, cavity_sizes))),
                dilated_betti_counts=dict(Counter(str([v["one_face_dilation"][f"beta{i}"] for i in range(3)]) for v in values)))
    save_json(args.report / "sensitivity.json", dict(summary=summary, cases=sensitivity))
    erosion_summary, shifted_summary = {}, {}
    for subset in ("train", "validation"):
        rows = [c for c in cases if c["split"] == subset]
        erosion_summary[subset], shifted_summary[subset] = {}, {}
        for radius in ("1.0", "2.0"):
            record = {}
            retained = {}
            for region in REGIONS:
                retained[region] = np.array([c["regions"][region]["erosion"][radius]["voxels"] / c["regions"][region]["nodes"] for c in rows])
                lcc = np.array([c["regions"][region]["erosion"][radius]["largest_fraction"] for c in rows])
                record[region] = dict(retained_fraction_quantiles=np.quantile(retained[region], [0, .05, .5, .95, 1]).tolist(),
                    lcc_fraction_quantiles=np.quantile(lcc, [0, .05, .5, .95, 1]).tolist(),
                    core_lcc_below_95pct_cases=int((lcc < .95).sum()))
            delta = retained["anterior"] - retained["posterior"]
            record["paired"] = dict(anterior_retention_greater_cases=int((delta > 0).sum()), cases=len(rows),
                anterior_minus_posterior_median=float(np.median(delta)))
            erosion_summary[subset][radius] = record
        for offset in (-6, -3, -1, 0, 1, 3, 6):
            selected = [r for r in shifted if r["split"] == subset and r["offset_slices"] == offset]
            shifted_summary[subset][str(offset)] = dict(cases=len(selected),
                anterior_more_compact_cases=sum(r["anterior_more_compact"] for r in selected))
    prediction_path = args.report / "prediction_cases.json"
    if prediction_path.exists():
        prediction_rows = json.loads(prediction_path.read_text())
        shifted_summary["existing_predictions"] = {}
        for arm, rows in prediction_rows.items():
            delta = [c["regions"]["anterior"]["degree_histogram"][6] / c["regions"]["anterior"]["nodes"]
                     - c["regions"]["posterior"]["degree_histogram"][6] / c["regions"]["posterior"]["nodes"] for c in rows]
            shifted_summary["existing_predictions"][arm] = dict(cases=len(rows),
                anterior_more_compact_cases=sum(d > 0 for d in delta), median_margin=float(np.median(delta)))
    save_json(args.report / "erosion_summary.json", erosion_summary)
    save_json(args.report / "compactness_probe.json", dict(summary=shifted_summary, cases=shifted))
    save_json(args.report / "verification.json", dict(graphs_verified=len(hashes),
        verifier_sha256=digest(__file__), generator_sha256=digest(source_path),
        checks=["Input and source SHA256", "Every node and semantic label", "Every edge and face axis",
                "No missing or duplicate edge", "Affine preserved", "Six-neighbor components",
                "Whole/region/cut edge decomposition", "Cavity counts"], graph_sha256=hashes))
    print(json.dumps(dict(graphs_verified=len(hashes), sensitivity=summary), indent=2))


if __name__ == "__main__":
    main()
