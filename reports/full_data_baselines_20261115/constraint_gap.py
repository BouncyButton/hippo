"""Compare one fixed low-data constraint per model against paired full-data runs."""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from tempfile import NamedTemporaryFile


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODELS = {"nnunet": "nnU-Net", "swinunetr": "SwinUNETR", "basic_unet16": "Basic U-Net",
          "unetr_vit_b16": "UNETR", "medsam3": "MedSAM3"}


def objectives(dataset, model, source):
    """Read the low-data matrix from the same execution batch as the paired control."""
    arms = defaultdict(dict)
    if source.suffix == ".json":
        study = dataset + ("_MedSAM3" if model == "medsam3" else "_capped")
        records = [r for r in json.loads(source.read_text())["rows"]
                   if r["study"] == study and r["model"].lower() == model]
    else:
        with source.open() as stream:
            records = [r for r in csv.DictReader(stream)
                       if r.get("model", model) == model and r.get("status", "verified") == "verified"]
    for r in records:
        arm = r.get("objective") or r["arm"]
        arm = arm.removeprefix("dice_")
        if arm in ("dice", "baseline"):
            continue
        if "dice" in r:
            score = float(r["dice"])
        elif "validation_dice_pct" in r:
            score = float(r["validation_dice_pct"])
        else:
            score = 100 * float(r.get("final_dice") or r["validation_dice"])
        pair = int(r["fold"]), int(r["seed"])
        assert pair not in arms[arm] and 0 <= score <= 100
        arms[arm][pair] = score
    return arms


def write_constraint_gaps(scores):
    groups = defaultdict(list)
    for row in scores:
        groups[row["dataset"], row["model"]].append(row)
    report = ["# How much of the full-data Dice gap do low-data constraints close?", "",
              "Scores are mean validation Dice (%). Each row compares the same folds, seeds and held-out cases in all three columns. N is the number of verified full-data runs out of nine.", "",
              "For each model and dataset, the constraint is the single combination with the highest mean over all nine archived low-data runs with a common baseline. This is a descriptive best observed on validation. The combination stays fixed as additional full-data results arrive. All combinations use the baseline Dice loss.", "",
              "The primary comparison uses each measured low-data training recipe. A separate table reports all 54 full-data runs that used peak LR 0.001. Those gains combine changes in data and learning rate. MSD and ADNI nnU-Net and SwinUNETR use the one-pass compute exception. The remaining groups use the original schedule.", "",
              "Full gain = full baseline − low baseline, in percentage points. Remaining = full baseline − constrained low data. A negative remaining value means the low-data constraint scored higher. Gap closed = 100 × (constrained low − low baseline) / full gain. It is reported only when full gain is positive. Values over 100% indicate the constraint surpassed the full baseline. A small full gain can produce a large percentage.", ""]
    header = ["| Dataset | Model | N | Best low-data constraint | Low | Low + constraint | Full | Full gain (pp) | Remaining (pp) | Gap closed |",
              "|---|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    sections = [[], []]
    deviations = []
    complete_groups = []
    for ds in ("MSD", "ADNI", "HFH", "MNI"):
        for model, title in MODELS.items():
            g = groups[ds, model]
            keys = {(int(r["partition"]), int(r["seed"])) for r in g}
            assert len(keys) == len(g) == 9
            source, = {r["source"] for r in g}
            selected, = {r["selected_arm"].removeprefix("dice_") for r in g}
            arms = objectives(ds, model, ROOT / source)
            eligible = {a: vals for a, vals in arms.items() if vals.keys() == keys}
            best = max(eligible, key=lambda a: (mean(eligible[a].values()), a == selected, a))
            for r in g:
                pair = int(r["partition"]), int(r["seed"])
                assert math.isclose(eligible[selected][pair], float(r["few_data_selected_pct"]), abs_tol=1e-8)
            if best != selected:
                deviations.append(f"| {ds} | {title} | {selected} | {mean(eligible[selected].values()):.2f} | {best} | {mean(eligible[best].values()):.2f} |")
            tuned = ds in ("ADNI", "HFH", "MSD") and model in ("basic_unet16", "unetr_vit_b16")
            if tuned:
                assert all(r["development_peak_lr_0_001_pct"] not in ("", None) for r in g)
            for ix in (0, 1):
                if ix and not tuned:
                    continue
                matched = []
                for r in g:
                    value = (r["development_peak_lr_0_001_pct"] if tuned and ix == 1 else
                             (r["paper_matched_full_pct"] if r["paper_matched_full_pct"] not in ("", None) else r["native_pass_full_pct"]))
                    if value not in ("", None):
                        matched.append((float(r["few_data_baseline_pct"]), eligible[best][int(r["partition"]), int(r["seed"])], float(value)))
                if not matched:
                    if not ix:
                        sections[ix].append(f"| {ds} | {title} | 0/9 | {best} | — | — | — | — | — | — |")
                    continue
                low, constrained, full = map(mean, zip(*matched))
                gain, remaining = full - low, full - constrained
                if ix == 0 and len(matched) == 9:
                    secondary = mean(float(r["development_peak_lr_0_001_pct"]) for r in g) if tuned else full
                    complete_groups.append((low, constrained, full, secondary))
                closed = f"{100 * (constrained - low) / gain:.1f}%" if gain > 0 else "—"
                patch = not ix and ds in ("MSD","ADNI") and model in ("nnunet","swinunetr")
                label = title + ("¹" if patch else "²" if tuned and ix == 1 else "³" if ds == "MSD" and model == "medsam3" else "")
                sections[ix].append(f"| {ds} | {label} | {len(matched)}/9 | {best} | {low:.2f} | {constrained:.2f} | {full:.2f} | {gain:+.2f} | {remaining:+.2f} | {closed} |")
    higher = [row for row in complete_groups if row[2] > row[0]]
    partial = sum(low < constrained < full for low, constrained, full, _ in higher)
    met = sum(constrained >= full for _, constrained, full, _ in higher)
    secondary_wins = sum(secondary > low for low, _, _, secondary in complete_groups)
    report.extend(["## Results across completed comparisons", "",
                   f"Full data scored higher in **{len(higher)} of {len(complete_groups)}** complete model–dataset comparisons using the primary recipes, including the labeled one-pass exceptions. Within those {len(higher)}, low-data constraints closed part of the gap in **{partial}** and matched or exceeded the full-data baseline in **{met}**. Substituting all six higher-learning-rate groups, {secondary_wins} of the same {len(complete_groups)} favored full data. Each completed model–dataset comparison receives one vote.", "",
                   "## Full-data comparison using the baseline training settings", "", *header, *sections[0], "",
                   "¹ These 36 full-size runs extend the original compute cap to cover every crop exactly once. The smaller-patch controls are available in the [separate results table](full_data_results.md#smaller-training-patches).", "",
                   "² Full-data peak LR 0.001. The complete groups used identical training splits, seeds, update budgets and initializations to their original full-data counterparts.", "",
                   "³ MSD MedSAM3 requires 208 training crops in each fold. The paired scores use the original 46, 52 or 44 held-out crops. The per-run CSV also reports all 52 validation crops. [MedSAM3 fine-tuning and crop coverage](full_data_results.md#medsam3-fine-tuning-audit)", "",
                   "HFH MedSAM3's measured low-data baseline and constraint used gain augmentation. The earlier nine full-data runs without augmentation averaged 69.85% against 69.06% for low data and 73.01% for low data with SDF + depth. The table uses only full-data runs with the same gain augmentation.", "",
                   "## Separate higher-learning-rate experiment", "",
                   "These six full-data groups used peak LR 0.001. Each low-data comparator retained its original learning rate.", "",
                   *header, *sections[1], "", "## Constraint selection", ""])
    if deviations:
        report.extend(["The following rows differ from the constraint selected in the project comparison. This selection table uses all nine low-data runs even when the primary table is interim.", "",
                       "| Dataset | Model | Project selection | Dice (%) | Best observed | Dice (%) |",
                       "|---|---|---|---:|---|---:|", *deviations, ""])
    else:
        report.extend(["The best observed constraint matches the project selection in all twenty comparisons.", ""])
    report.extend(["MSD MedSAM3 uses constraints from the execution batch of its matched low-data baseline. Two additional constraints from the original batch had a different baseline. HFH includes combinations with nine verified low-data results. The constrained scores in these tables all come from low-data training.", "",
                   "The archived MNI MedSAM3 design selected seeds using validation outcomes, including constraint gains. This comparison retains those seeds. [Seed selection, training exposure and error analysis](hfh_pilot_diagnosis.md)", "",
                   "[Primary results and protocol details](full_data_results.md) · [Paired trial data and archive paths](paired_comparisons.csv)", ""])
    with NamedTemporaryFile(mode="w", dir=HERE, delete=False) as stream:
        stream.write("\n".join(report))
        path = Path(stream.name)
    path.replace(HERE / "constraint_gap.md")


if __name__ == "__main__":
    with (HERE / "paired_comparisons.csv").open() as stream:
        write_constraint_gaps(csv.DictReader(stream))
