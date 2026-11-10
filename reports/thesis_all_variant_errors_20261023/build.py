"""Reconstruct full-validation errors from retained, verified evaluations.

This script reads local archives only. It does not run models or retrieve data.
Source campaign identifiers are retained for provenance, not thesis terminology.
"""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RECOVERY = ROOT / "reports/background_dice_archive_recovery_20261023"
MEDICAL = ROOT / "reports/thesis_noaug_update_20261021/results.json"
GEO = ROOT / "reports/thesis_geofwi_update_20261022/results.json"
ARMS = ["baseline", "bands", "sdf", "bands_sdf", "bands_edge",
        "bands_edge_sdf", "depth", "bands_depth", "edge_depth", "sdf_depth",
        "bands_edge_depth", "bands_sdf_depth", "edge_sdf_depth",
        "bands_edge_sdf_depth"]
MODELS = ["nnunet", "swinunetr", "unetr_vit_b16", "basic_unet16"]
MODEL_NAMES = dict(zip(MODELS, ["nnU-Net", "SwinUNETR", "UNETR", "3D U-Net"]))
INVENTORIES = {
    "GeoFWI3D_conventional": ({0, 1, 2}, {0, 1, 2}),
    "MSD_conventional": ({0, 1, 2}, {0, 2, 20}),
    "ADNI_conventional": ({0, 1, 2}, {0, 1, 4}),
    "GeoFWI3D_SAM3": ({0, 1, 3}, {0, 1, 2}),
    "MSD_MedSAM3_extension": ({0, 1, 3}, {17, 83, 191}),
    "MSD_MedSAM3_original": ({0, 1, 3}, {17, 83, 191}),
    "ADNI_MedSAM3": ({0, 1, 3}, {0, 1, 2}),
}


def normalize_arm(arm):
    return "baseline" if arm in {"dice", "baseline"} else arm.removeprefix("dice_")


def label(arm):
    symbols = {"bands": "B", "edge": "E", "sdf": "S", "depth": "D"}
    return "Baseline" if arm == "baseline" else "Baseline $+" + "+".join(symbols[x] for x in arm.split("_")) + "$"


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build():
    metadata = read_csv(RECOVERY / "background_dice_runs.csv")
    by_run = {(x["campaign"], x["run_id"]): x for x in metadata}
    assert len(by_run) == len(metadata) == 1862
    totals = defaultdict(lambda: {"cases": 0, "fn": 0, "fp": 0, "swaps": 0})
    seen = set()
    matrices = 0
    with (RECOVERY / "background_dice_cases.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = row["campaign"], row["run_id"]
            identity = (*key, row["case"])
            assert identity not in seen, identity
            seen.add(identity)
            meta = by_run[key]
            for field in ["fold", "seed", "model", "arm"]:
                assert row[field] == meta[field], (identity, field)
            cm = json.loads(row["confusion_json"])
            n = int(row["n_classes"])
            assert n == (3 if row["campaign"].startswith("MSD") else 2)
            assert len(cm) == n and all(len(a) == n for a in cm)
            assert all(isinstance(v, int) and v >= 0 for a in cm for v in a)
            assert sum(map(sum, cm)) == int(row["voxel_count"])
            counts = totals[key]
            counts["cases"] += 1
            counts["fn"] += sum(cm[c][0] for c in range(1, n))
            counts["fp"] += sum(cm[0][c] for c in range(1, n))
            counts["swaps"] += cm[1][2] + cm[2][1] if n == 3 else 0
            matrices += 1
    assert matrices == 124860

    runs = []
    for key, meta in by_run.items():
        counts = totals[key]
        assert counts["cases"] == int(meta["validation_cases"])
        source = ROOT / meta["evaluation_source"]
        assert digest(source) == meta["evaluation_sha256"], source
        pooled = meta["campaign"].endswith("conventional")
        divisor = 1 if pooled else counts["cases"]
        runs.append({
            "campaign": key[0], "run_id": key[1], "model": meta["model"],
            "arm": normalize_arm(meta["arm"]), "fold": int(meta["fold"]),
            "seed": int(meta["seed"]), "validation_cases": counts["cases"],
            "unit": "pooled validation voxels per run" if pooled else "mean validation voxels per volume",
            "fn": counts["fn"] / divisor, "fp": counts["fp"] / divisor,
            "swaps": counts["swaps"] / divisor if key[0].startswith("MSD") else None,
            "source": meta["evaluation_source"], "sha256": meta["evaluation_sha256"],
        })

    medical = json.loads(MEDICAL.read_text())
    campaign_map = {"MSD_capped": "MSD_conventional", "ADNI_capped": "ADNI_conventional",
                    "MSD_MedSAM3": "MSD_MedSAM3_extension", "ADNI_MedSAM3": "ADNI_MedSAM3"}
    index = {(x["campaign"], x["model"], x["arm"], x["fold"], x["seed"]): x for x in runs}
    checks = Counter()
    for row in medical["rows"]:
        if row["study"] == "MSD_MedSAM3_original":
            # These 27 records preserve aggregate errors, but no case matrices.
            runs.append({
                "campaign": row["study"], "run_id": row["source_id"],
                "model": row["model"], "arm": row["arm"],
                "fold": row["fold"], "seed": row["seed"],
                "validation_cases": {0: 46, 1: 52, 3: 44}[row["fold"]],
                "unit": row["error_unit"],
                **{metric: row[metric] for metric in ["fn", "fp", "swaps"]},
                "source": str(MEDICAL.relative_to(ROOT)), "sha256": digest(MEDICAL),
            })
            checks["preserved_aggregate_records"] += 1
            continue
        key = (campaign_map[row["study"]], row["model"], row["arm"], row["fold"], row["seed"])
        derived = index[key]
        for metric in ["fn", "fp", "swaps"]:
            expected, actual = row[metric], derived[metric]
            assert (expected is None and actual is None) or abs(expected - actual) < 1e-7, (key, metric, expected, actual)
        checks["medical_runs_matching_previous_error_extraction"] += 1
    geo = json.loads(GEO.read_text())
    for row in geo["rows"]:
        key = ("GeoFWI3D_conventional", row["model"], normalize_arm(row["arm"]), row["fold"], row["seed"])
        for metric in ["fn", "fp"]:
            expected = row.get("validation_" + metric)
            if expected is not None:
                assert abs(expected - index[key][metric]) < 1e-7, (key, metric, expected, index[key][metric])
                checks["geological_error_values_matching_previous_extraction"] += 1

    groups = defaultdict(list)
    for row in runs:
        groups[row["campaign"], row["model"], row["arm"]].append(row)
    conditions = []
    for (campaign, model, arm), rows in sorted(groups.items()):
        folds, seeds = INVENTORIES[campaign]
        expected = {(f, s) for f in folds for s in seeds}
        present = {(x["fold"], x["seed"]) for x in rows}
        assert len(present) == len(rows) and present <= expected, (campaign, model, arm, present)
        complete = present == expected
        condition = {"campaign": campaign, "model": model, "arm": arm,
                     "runs": len(rows), "complete": complete,
                     "missing_cells": sorted(expected - present), "unit": rows[0]["unit"]}
        for metric in ["fn", "fp", "swaps"]:
            if complete and all(x[metric] is not None for x in rows):
                fold_means = {str(f): mean(x[metric] for x in rows if x["fold"] == f) for f in sorted(folds)}
                condition[metric] = mean(fold_means.values())
                condition[metric + "_fold_sd"] = stdev(fold_means.values())
                condition[metric + "_fold_means"] = fold_means
            else:
                condition[metric] = condition[metric + "_fold_sd"] = None
                condition[metric + "_fold_means"] = None
        conditions.append(condition)
    assert len(conditions) == 211 and sum(x["complete"] for x in conditions) == 203
    # Reproduce all previously reported medical condition means as well.
    ci = {(x["campaign"], x["model"], x["arm"]): x for x in conditions}
    for old in medical["summaries"]:
        key = (campaign_map.get(old["study"], old["study"]), old["model"], old["arm"])
        for metric in ["fn", "fp", "swaps"]:
            if old[metric] is not None:
                assert abs(old[metric] - ci[key][metric]) < 1e-7, (key, metric)
    audit = {
        "matrix_records_checked": matrices, "evaluation_file_hashes_checked": len(metadata),
        "coverage": dict(Counter(x["campaign"] for x in runs)), "checks": dict(checks),
        "condition_rows_including_two_msd_controls": len(conditions),
        "complete_conditions": sum(x["complete"] for x in conditions),
        "sam3_snapshot": "23 October 2026, 13:37 UTC; not a live completion check",
        "reduction": "Conventional: pool cases within run. Adaptation: mean per volume within run. Then average seeds within fold and folds equally; sample SD across three fold means. Incomplete conditions have no aggregate.",
        "error_definition": "Reference rows, predicted columns. FN=sum C[c,0]; FP=sum C[0,c], c>0. MSD A/P swaps=C[1,2]+C[2,1]. Binary swaps not applicable.",
        "input_sha256": {str(p.relative_to(ROOT)): digest(p) for p in [RECOVERY / "background_dice_cases.csv", RECOVERY / "background_dice_runs.csv", MEDICAL, GEO]},
        "issues": [],
    }
    for name, value in [("run_errors.json", runs), ("condition_errors.json", conditions), ("verification.json", audit)]:
        (OUT / name).write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    return conditions


if __name__ == "__main__":
    build()
