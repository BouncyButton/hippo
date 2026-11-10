"""Reproduce the matched-epoch comparison from archived aggregate records only."""
from pathlib import Path
import csv
import hashlib
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/hippo-progression-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
AUDIT = OUT.parent / "final_audit.json"
CAP_OLD = ROOT / "reports/capped_replication_31976_20261007/final_summary_20261010.json"
CAP_NEW = ROOT / "reports/capped_winner_31976_20261013/final_results_20261013.json"
records = {r["id"]: r for r in json.loads(AUDIT.read_text())["records"]}
curves = {
    key: {int(c["epoch"]): float(c["validation_dice"]) for c in r["curves"]}
    for key, r in records.items()
}
FOCUS = "winning_constraint"
ALL = "dice_bands_edge_sdf_depth"
EDGE = "dice_bands_edge"
names = {
    FOCUS: "Focused bands + endpoints + depth CE",
    ALL: "Original bands + edge + SDF + depth CE",
    EDGE: "Original bands + edge",
    "dice": "Dice only",
}
colors = {FOCUS: "#1864AB", ALL: "#D16A15", EDGE: "#278268", "dice": "#8A9199"}

# Verify epoch counts, common warmup, and selected bests before plotting.
for key, curve in curves.items():
    completion = records[key]["completion"]
    assert list(curve) == list(range(1, completion["completed_epochs"] + 1))
    assert max(curve, key=curve.get) == completion["best_epoch"]
    assert all(curve[e] == curves[FOCUS][e] for e in range(1, 6))

rows = []
for key, r in records.items():
    best = 0.0
    for c in r["curves"]:
        score = float(c["validation_dice"])
        best = max(best, score)
        rows.append({"id": key, "epoch": int(c["epoch"]), "updates": int(c["updates"]),
                     "learning_rate": float(c["learning_rate"]),
                     "validation_dice_percent": score * 100,
                     "best_so_far_percent": best * 100})
with (OUT / "all_matched_epoch_curves.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

old_data = json.loads(CAP_OLD.read_text())
new_data = json.loads(CAP_NEW.read_text())
old_runs = {(int(r["fold"]), int(r["seed"])): r
            for r in old_data["runs"] + old_data["reused_runs"]
            if r["model"] == "swinunetr" and r["objective"] == ALL}
new_runs = {(int(r["fold"]), int(r["seed"])): r for r in new_data["runs"]
            if r["model"] == "swinunetr" and r["objective"] == "dice_focus_bands_pair_depth"}
assert set(old_runs) == set(new_runs) and len(old_runs) == 9
assert old_runs[(0, 0)]["updates"] == new_runs[(0, 0)]["updates"] == 206

focus, old, edge = (curves[key] for key in (FOCUS, ALL, EDGE))
shared = sorted(set(focus) & set(old))
lead_epochs = [e for e in shared if old[e] > focus[e]]
sustained_lead = next(e for e in shared if all(focus[t] > old[t] for t in shared if t >= e))
summary = {
    "scope": "SwinUNETR, 5% training data, fold 0, seed 0; retrospective validation analysis",
    "sources_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in [AUDIT, CAP_OLD, CAP_NEW]},
    "old_capped_recipe_leads_focused_at_epochs": lead_epochs,
    "focused_sustained_lead_from_epoch": sustained_lead,
    "max_early_old_lead_pp": max((old[e] - focus[e]) * 100 for e in shared),
    "focused_exceeds_all_original_contenders_final_bests_at_epoch": next(
        e for e in focus if focus[e] > max(max(c.values()) for k, c in curves.items() if k != FOCUS)),
    "cap_update_equivalent_epochs": 20.6,
    "cap_bracketing_epochs": [20, 21],
    "exact_206_update_validation_in_long_run": False,
    "best_and_stop": {
        key: {"best_dice_percent": max(curves[key].values()) * 100,
              "best_epoch": records[key]["completion"]["best_epoch"],
              "completed_epochs": records[key]["completion"]["completed_epochs"],
              "early_stopping": records[key]["completion"]["early_stopping"]}
        for key in names},
    "actual_capped_comparison": {
        "fold0_seed0_old_final_percent": old_runs[(0, 0)]["final_dice"] * 100,
        "fold0_seed0_new_final_percent": new_runs[(0, 0)]["final_dice"] * 100,
        "fold0_seed0_old_best_percent": old_runs[(0, 0)]["best_dice"] * 100,
        "fold0_seed0_new_best_percent": new_runs[(0, 0)]["best_dice"] * 100,
        "old_nine_run_mean_percent": np.mean([r["final_dice"] for r in old_runs.values()]) * 100,
        "new_nine_run_mean_percent": np.mean([r["final_dice"] for r in new_runs.values()]) * 100,
        "new_paired_wins_of_nine": sum(new_runs[k]["final_dice"] > old_runs[k]["final_dice"] for k in old_runs),
    },
    "limitations": [
        "Equal update count is not exactly equal FLOPs: objective arithmetic differs.",
        "Capped runs used mild_v1 augmentation and 31-update LR warmup then a 610-update cosine horizon; long runs used no augmentation and StepLR every 20 epochs.",
        "The old capped all-four recipe activated SDF at update 116; the long-run contender included SDF in the common update-51-to-100 constraint ramp. Recalibrated coefficients also differ.",
        "Capped headline comparisons use final-budget checkpoints; long-run headline comparisons select best validation checkpoints.",
        "This exploratory fold/seed was used to select the focused constraint. Results do not establish ranking on independent data.",
        "Early stopping does not establish what would happen if training were continued beyond stopping.",
    ],
}
(OUT / "analysis.json").write_text(json.dumps(summary, indent=2) + "\n")

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, (ax, zoom) = plt.subplots(1, 2, figsize=(14.2, 6.5), gridspec_kw={"width_ratios": [1.7, 1]})
fig.subplots_adjust(left=.055, right=.98, bottom=.26, top=.84, wspace=.23)
fig.suptitle("The focused constraint overtakes early and reaches its best sooner", x=.055,
             y=.97, ha="left", fontsize=18, fontweight="bold")
fig.text(.055, .913, "SwinUNETR · 5% data · fold 0 / seed 0 · identical 75-epoch maximum protocol",
         fontsize=11.5, color="#515962")
for key in names:
    curve = curves[key]
    xs = list(curve)
    ys = np.array(list(curve.values())) * 100
    ax.plot(xs, ys, color=colors[key], lw=2.4 if key == FOCUS else 1.7, label=names[key])
    best_epoch = records[key]["completion"]["best_epoch"]
    ax.scatter(best_epoch, curve[best_epoch] * 100, marker="*", s=125, color=colors[key],
               edgecolors="white", linewidths=.6, zorder=5)
    ax.scatter(xs[-1], ys[-1], marker="x", s=35, color=colors[key], zorder=5)
ax.axvspan(20, 21, color="#B4BEC8", alpha=.3, zorder=0)
ax.text(22, 11, "206 updates = 20.6 epochs\n(cap-equivalent update count)", fontsize=9, color="#515962")
ax.annotate("81.502% at epoch 41", xy=(41, focus[41] * 100), xytext=(34, 88),
            fontsize=10, color=colors[FOCUS], arrowprops={"arrowstyle": "-", "color": colors[FOCUS]})
ax.set(xlim=(1, 76), ylim=(0, 92), xlabel="Completed training epochs", ylabel="Validation Dice (%)",
       title="Recorded curves; lines end when training ends")
ax.set_xticks([1, 10, 20, 30, 40, 50, 60, 75])
ax.grid(axis="y", alpha=.15)
ax.legend(loc="upper left", bbox_to_anchor=(-.01, -.19), frameon=False, fontsize=10, ncol=1)

xs = np.arange(6, 22)
diff = np.array([(focus[int(e)] - old[int(e)]) * 100 for e in xs])
zoom.axhline(0, lw=1, color="#656D75")
zoom.fill_between(xs, 0, diff, where=diff >= 0, interpolate=True, color=colors[FOCUS], alpha=.15)
zoom.fill_between(xs, 0, diff, where=diff <= 0, interpolate=True, color=colors[ALL], alpha=.22)
zoom.plot(xs, diff, color="#3F4B59", lw=1.8, marker="o", ms=3)
zoom.axvspan(20, 21, color="#B4BEC8", alpha=.25, zorder=0)
zoom.annotate("Old recipe +4.047 pp", xy=(12, diff[6]), xytext=(6.4, -6.1),
              fontsize=9.5, color=colors[ALL], arrowprops={"arrowstyle": "-", "color": colors[ALL]})
zoom.annotate("Focused takes the lead\nat epoch 14", xy=(14, diff[8]), xytext=(6.5, 7.8),
              fontsize=9.5, color=colors[FOCUS], arrowprops={"arrowstyle": "-", "color": colors[FOCUS]})
zoom.set(xlim=(5.5, 21.6), ylim=(-7.3, 12), xlabel="Completed training epochs",
         ylabel="Focused minus old all-four recipe (pp)", title="Early lead switches at epoch 14")
zoom.set_xticks([6, 8, 10, 12, 14, 16, 18, 20, 21])
zoom.grid(axis="y", alpha=.15)
fig.text(.65, .11, "Stars: best checkpoints. Crosses: last epochs.\nAll plotted runs: no augmentation, LR halved every 20 epochs.\nThe actual capped runs used a different training recipe.\nNo interpolation of validation at update 206.",
         fontsize=9.3, color="#515962", va="center")
fig.savefig(OUT / "validation_trajectories.png", dpi=180)
fig.savefig(OUT / "validation_trajectories.pdf")
plt.close(fig)

lines = ["# Validation trajectories: capped winner versus longer-run winner", "",
         summary["scope"] + ".", "",
         "The old capped-table winner is Dice + original bands + edge + signed distance + depth CE. "
         "The focused recipe is Dice + focused bands + endpoints + depth CE. All table entries below "
         "are observed epoch-end validation scores, not running maxima.", "",
         "| Epoch | Focused recipe | Original all four | Original bands + edge | Dice only |",
         "|---:|---:|---:|---:|---:|"]
for e in [10, 12, 13, 14, 15, 20, 21, 30, 41, 49, 58, 66, 69, 75]:
    values = [f"{curves[k][e] * 100:.3f}%" if e in curves[k] else "—" for k in names]
    lines.append(f"| {e} | " + " | ".join(values) + " |")
lines += ["", "The old recipe leads at epochs 10–13; its largest lead is 4.047 percentage points "
          "at epoch 12. The focused recipe overtakes it at epoch 14 and stays ahead throughout "
          "the remaining shared epochs. At epoch 18, focused Dice already exceeds the eventual "
          "best checkpoint of every original contender.", "",
          "The capped experiment used 206 updates (20.6 patient-pass epochs). The longer experiment "
          "has recorded validation at 200 and 210 updates, so epochs 20 and 21 bracket this point. "
          "They are update-matched approximations, not exact FLOPs-matched measurements.", "",
          "| Recipe | Best validation Dice | Best epoch | Last epoch | Stopped by patience? |",
          "|---|---:|---:|---:|---|"]
for k in names:
    s = summary["best_and_stop"][k]
    lines.append(f"| {names[k]} | {s['best_dice_percent']:.3f}% | {s['best_epoch']} | "
                 f"{s['completed_epochs']} | {'Yes' if s['early_stopping']['triggered'] else 'No: epoch limit'} |")
lines += ["", "The focused recipe reaches its best and stops earlier. The hypothesis that it "
          "wins only because it trains longer than the old all-four recipe is not supported. "
          "Early stopping used minimum 25 epochs, patience 8, and minimum improvement 0.0005 "
          "(0.05 percentage points); exact best-checkpoint selection was separate.", "",
          "Among the original contenders, there is a smaller speed/final-score tradeoff: at epoch 20 "
          "the all-four recipe leads bands + edge by 6.617 percentage points. They become close "
          "around epochs 30–40 and trade places. The all-four best is 77.204% at epoch 58; bands + "
          "edge ultimately reaches 77.303% at epoch 69, after the all-four run stopped at 66. "
          "That 0.099-point difference is descriptive evidence from one tuned fold/seed, not proof "
          "of a reproducible ranking.", "", "## Actual capped results and protocol differences", ""]
c = summary["actual_capped_comparison"]
lines += [f"Under the actual cap, fold 0 / seed 0 also favors the focused recipe: "
          f"{c['fold0_seed0_new_final_percent']:.3f}% versus {c['fold0_seed0_old_final_percent']:.3f}% "
          f"at the final budget checkpoint. Selecting each capped run's best recorded checkpoint "
          f"gives {c['fold0_seed0_new_best_percent']:.3f}% versus {c['fold0_seed0_old_best_percent']:.3f}%.", "",
          f"Across all nine capped fold/seed runs, the focused mean is {c['new_nine_run_mean_percent']:.3f}% "
          f"versus {c['old_nine_run_mean_percent']:.3f}%, but focused wins only "
          f"{c['new_paired_wins_of_nine']}/9 paired comparisons. The original all-four recipe therefore "
          "wins more individual pairs despite having the slightly lower mean.", "",
          "Capped training used mild_v1 augmentation, a 31-update LR warmup and a fixed 610-update "
          "cosine horizon. The longer comparison used no augmentation, initial LR 1e-4, and "
          "halving every 20 epochs. The capped old recipe delayed SDF to update 116; the long run "
          "included it in the shared update-51-to-100 constraint ramp. Calibration coefficients "
          "were also recalculated. Thus these are not the same trajectory stopped at different "
          "times, and the available records cannot separate the causal effects of augmentation, "
          "schedule, calibration, and SDF timing.", "",
          "To isolate training duration, a future experiment would hold all of those settings "
          "fixed, validate exactly at update 206, and continue the same trajectories to later "
          "checkpoints. A factorial follow-up would be needed to attribute effects to individual "
          "schedule/augmentation changes. No training was launched for this analysis.", "",
          "## Reproduction and provenance", "",
          "Run `.venv/bin/python reports/swin_contenders75_20261013/trajectory_analysis/analyze.py` "
          "from the repository root. `analysis.json` records source hashes and derived results; "
          "`all_matched_epoch_curves.csv` contains all 15 trajectories and their running bests. "
          "The plot is available as PNG and PDF. These exports contain aggregate scores, not patient images.", "",
          "Additional limitations:", ""]
lines += ["- " + x for x in summary["limitations"]]
(OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
print(json.dumps(summary, indent=2))
