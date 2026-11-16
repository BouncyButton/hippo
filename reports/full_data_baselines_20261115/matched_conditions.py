"""Report complete baseline-recipe pairs, with one row for each fold and seed."""
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean
from tempfile import NamedTemporaryFile


DEST = Path(__file__).resolve().parent
ROOT = DEST.parents[1]
MODELS = {"medsam3":"MedSAM3", "nnunet":"nnU-Net", "swinunetr":"SwinUNETR",
          "basic_unet16":"Basic U-Net", "unetr_vit_b16":"UNETR"}


def write_matched_conditions(scores):
    groups = defaultdict(list)
    for key, row in scores.items():
        if row["paper_matched_full_pct"] is not None:
            groups[row["dataset"], row["model"]].append((key, row))
    count = sum(len(group) for group in groups.values())
    lines = ["# Full-data baselines with the low-data training recipe", "",
             f"{count} verified fold/seed pairs across {len(groups)} dataset/model combinations. Each paired comparison uses the same seed, validation cases, baseline loss, training resolution, augmentation policy, learning-rate schedule and optimizer update count. The full-data sampler covers every eligible training crop. Increasing the training pool changes the number of times each crop is sampled.", "",
             "One run produces one mean validation Dice across its held-out cases. The group mean gives equal weight to each fold/seed run. The constrained column is the low-data constraint combination with the highest nine-run mean on validation. The same combination is used for all nine runs of that group. Differences use unrounded Dice and are expressed in percentage points.", "",
             "| Dataset | Model | Pairs | Folds | Seeds | Updates/run | Low | Full | Full − low | Best constrained low | Constraint |",
             "|---|---|---:|---|---|---:|---:|---:|---:|---:|---|"]
    details = []
    for ds in ("MSD","ADNI","HFH","MNI"):
        for model, title in MODELS.items():
            group = sorted(groups.get((ds, model), []), key=lambda item:(item[1]["partition"],item[1]["seed"]))
            if not group:
                continue
            rows = [r for _,r in group]
            folds = ", ".join(map(str, sorted({r["partition"] for r in rows})))
            seeds = ", ".join(map(str, sorted({r["seed"] for r in rows})))
            updates, = {r["optimizer_update_budget"] for r in rows}
            constraint, = {r["selected_arm"].removeprefix("dice_") for r in rows}
            low, full, best = (mean(r[k] for r in rows) for k in ("few_data_baseline_pct","paper_matched_full_pct","few_data_selected_pct"))
            lines.append(f"| {ds} | {title} | {len(rows)}/9 | {folds} | {seeds} | {updates} | {low:.2f} | {full:.2f} | {full-low:+.2f} | {best:.2f} | {constraint} |")
            details.extend([f"### {ds} · {title}", "", "| Fold | Seed | Updates | Training crops | Low Dice | Full Dice | Full − low | Constrained low Dice | Full-data archive |",
                            "|---:|---:|---:|---:|---:|---:|---:|---:|---|"])
            for key,r in group:
                archive = os.path.relpath(ROOT/r["paper_matched_archive"],DEST)
                lo,fu,co = (r[k] for k in ("few_data_baseline_pct","paper_matched_full_pct","few_data_selected_pct"))
                details.append(f"| {r['partition']} | {r['seed']} | {updates} | {r['eligible_training_crops']} | {lo:.2f} | {fu:.2f} | {fu-lo:+.2f} | {co:.2f} | [{key}]({archive}) |")
            details.append("")
    lines.extend(["","## Evidence and limits","",
                  "Scratch runs were checked against the actual low-data configurations, logged learning rates at their validation steps, final Dice, and per-case ground-truth class counts. The MNI scratch archives document the model, initialization recipe and seed. They lack a saved initial-weight fingerprint. The other matched scratch archives also confirm initial-weight fingerprints.","",
                  "MedSAM3 uses parameter-efficient LoRA fine-tuning. On completed ADNI and MNI pairs the three arms share the same initial adapter and frozen-backbone fingerprints. The full-data adapter changed during training, and the frozen backbone retained its hash. Dice was independently reconstructed from each held-out case's confusion matrix. MNI MedSAM3's historical seeds were chosen using validation outcomes.","",
                  "The MSD and ADNI nnU-Net and SwinUNETR one-pass experiments required extra optimizer updates. The separate peak-learning-rate experiment changed the learning-rate schedule. The earlier HFH MedSAM3 runs omitted gain augmentation. The historical MSD MedSAM3 runs used restricted training pools. These experiments are documented in the [broader experiment report](full_data_results.md).","",
                  "## Individual verified runs","",*details])
    with NamedTemporaryFile(dir=DEST,mode="w",delete=False) as stream:
        stream.write("\n".join(lines))
        path=Path(stream.name)
    path.replace(DEST/"matched_conditions.md")
