"""Report every declared low-data objective against its matched validation control."""
import argparse
import csv
import json
import math
from collections import defaultdict
from functools import lru_cache
from io import StringIO
from pathlib import Path
from statistics import mean
from tempfile import NamedTemporaryFile


HERE = Path(__file__).resolve().parent
ROOT = HERE / "data"
DATASETS = ("MSD", "ADNI", "MNI", "HFH")
MODELS = ("basic_unet16", "nnunet", "unetr_vit_b16", "swinunetr", "medsam3")
ARMS = ("baseline", "bands", "sdf", "bands_sdf", "bands_edge", "bands_edge_sdf",
        "depth", "bands_depth", "edge_depth", "sdf_depth", "bands_edge_depth",
        "bands_sdf_depth", "edge_sdf_depth", "bands_edge_sdf_depth")
FULL_CONTROLS = ("paper_matched_full_pct", "native_pass_full_pct", "development_peak_lr_0_001_pct")


# Recorded cohorts, not prospectively selected test seeds. Do not infer membership
# from whichever results happen to be available at reporting time.
SCRATCH_SEEDS = {"MSD": (0, 2, 20), "ADNI": (0, 1, 4), "MNI": (0, 5, 19), "HFH": (0, 1, 2)}


def cohort(dataset, model):
    folds = (0, 1, 2)
    seeds = SCRATCH_SEEDS[dataset]
    if model == "medsam3":
        folds = (0, 1, 3) if dataset in ("MSD", "ADNI") else folds
        seeds = (17, 83, 191) if dataset == "MSD" else (0, 1, 2)
    return {(fold, seed) for fold in folds for seed in seeds}


def percentage(value, scale=1):
    score = scale * float(value)
    if not math.isfinite(score) or not 0 <= score <= 100:
        raise ValueError(f"Invalid Dice percentage: {score}")
    return score


def read_rows(path):
    with path.open() as stream:
        return json.load(stream)["rows"] if path.suffix == ".json" else list(csv.DictReader(stream))


def objective_name(row):
    arm = row.get("objective") or row["arm"]
    return "baseline" if arm == "dice" else arm.removeprefix("dice_")


def index_source(records, dataset, model, pairs, paper_json):
    """Keep the original MSD adapter batch distinct from its later controls."""
    study = dataset + ("_MedSAM3" if model == "medsam3" else "_capped")
    primary, original = {}, {}
    for row in records:
        if row.get("model", model).lower() != model:
            continue
        if paper_json and row["study"] != study and not (
                study == "MSD_MedSAM3" and row["study"] == study + "_original"):
            continue
        target = original if paper_json and row["study"] != study else primary
        arm = objective_name(row)
        pair = int(row["fold"]), int(row["seed"])
        key = arm, pair
        if arm not in ARMS or pair not in pairs or key in target:
            raise ValueError(f"Unexpected or duplicate result: {dataset}, {model}, {key}")
        value = None
        if row.get("status", "verified") == "verified":
            if paper_json:
                value = percentage(row["dice"])
            elif "validation_dice_pct" in row:
                value = percentage(row["validation_dice_pct"])
            else:
                field = "final_dice" if "final_dice" in row else "validation_dice"
                value = percentage(row[field], 100)
        target[key] = value
    return primary, original


def comparison_rows(scores, root):
    groups = defaultdict(dict)
    for row in scores:
        group = row["dataset"], row["model"]
        pair = int(row["partition"]), int(row["seed"])
        if group[0] not in DATASETS or group[1] not in MODELS or pair in groups[group]:
            raise ValueError(f"Unexpected or duplicate control: {group}, {pair}")
        groups[group][pair] = row

    load = lru_cache(maxsize=None)(read_rows)
    for dataset in DATASETS:
        for model in MODELS:
            controls = groups[dataset, model]
            if controls.keys() != cohort(dataset, model):
                raise ValueError(f"Controls differ from the archived cohort for {dataset}, {model}")
            sources = {r["source"] for r in controls.values()}
            if len(sources) != 1:
                raise ValueError(f"Ambiguous result source for {dataset}, {model}")
            source = root / sources.pop()
            primary, original = index_source(load(source), dataset, model, controls, source.suffix == ".json")
            for arm in ARMS:
                # Only these two constrained objectives used the original MSD adapter batch.
                old_batch = (dataset, model) == ("MSD", "medsam3") and arm in ("bands", "bands_edge")
                values = original if old_batch else primary
                if old_batch and any(a == arm for a, _ in primary):
                    raise ValueError(f"Ambiguous execution batch for {arm}")
                for (fold, seed), control in sorted(controls.items()):
                    pair = fold, seed
                    value = values.get((arm, pair))
                    low = values.get(("baseline", pair))
                    if not old_batch and low is not None and not math.isclose(
                            low, percentage(control["few_data_baseline_pct"]), abs_tol=1e-8):
                        raise ValueError(f"Mismatched baseline: {dataset}, {model}, {pair}")
                    yield dict(dataset=dataset, model=model, objective=arm, fold=fold, seed=seed,
                               batch="original" if old_batch else "primary",
                               baseline_pct=low, objective_pct=value,
                               gain_pp=None if low is None or value is None else value - low,
                               **{k: percentage(control[k]) if not old_batch and control[k] not in
                                  ("", None) else None for k in FULL_CONTROLS})


def atomic_write(path, text):
    with NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        stream.write(text)
    Path(stream.name).replace(path)


def display(values, signed=True):
    return format(mean(values), "+.2f" if signed else ".2f") if len(values) == 9 else "—"


def write_constraint_gaps(scores, root=ROOT, output=HERE):
    rows = list(comparison_rows(scores, root))
    groups = defaultdict(list)
    for row in rows:
        groups[row["dataset"], row["model"], row["objective"]].append(row)

    table = ["# Validation results for every spatial objective", "",
             "Dice is in percent. Gains are paired differences in percentage points. "
             "A mean requires all nine fold–seed pairs; see the CSV for individual and missing cells. "
             "No objective is selected by its score in this report.", "",
             "These are historical development cohorts: the paper highlighted validation winners, "
             "and some seeds and recipes were chosen using validation outcomes. Showing the whole "
             "matrix does not make these independent test estimates.", "",
             "Full minus objective is shown separately for the matched recipe, the one-pass budget "
             "exception, and the higher-learning-rate experiment. A dash means incomplete or inapplicable. "
             "MSD MedSAM3 BAND and BAND+EDGE use their original batch baseline; no matching full-data "
             "control is assigned to those rows.", "",
             "| Dataset | Model | Objective | Paired cells | Objective Dice | Gain | Full minus objective: matched / one-pass / higher LR |",
             "|---|---|---|---:|---:|---:|---|"]
    for (dataset, model, arm), cells in groups.items():
        paired = [r for r in cells if r["gain_pp"] is not None]
        gaps = [display([r[k] - r["objective_pct"] for r in paired if r[k] is not None])
                for k in FULL_CONTROLS]
        table.append(f"| {dataset} | {model} | {arm} | {len(paired)}/9 | "
                     f"{display([r['objective_pct'] for r in paired], False)} | "
                     f"{display([r['gain_pp'] for r in paired])} | {' / '.join(gaps)} |")
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=rows[0].keys(), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    output.mkdir(parents=True, exist_ok=True)
    atomic_write(output / "objective_pairs.csv", buffer.getvalue())
    atomic_write(output / "constraint_gap.md", "\n".join(table) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls", type=Path, default=ROOT / "controls.csv")
    parser.add_argument("--root", type=Path, default=ROOT, help="Root for archived result source paths")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.controls.open() as stream:
        write_constraint_gaps(csv.DictReader(stream), args.root, args.output)
