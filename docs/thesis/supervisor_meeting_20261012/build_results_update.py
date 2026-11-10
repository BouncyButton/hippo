"""Verify and plot the completed 13 October replication and FLOP transfer."""
from pathlib import Path
import hashlib
import json
import os
import statistics as st

os.environ.setdefault("MPLCONFIGDIR", "/tmp/hippo-supervisor-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FIG = HERE / "figures"
MODELS = [("basic_unet16", "3D U-Net width 16"), ("swinunetr", "SwinUNETR"),
          ("nnunet", "nnU-Net"), ("unetr_vit_b16", "UNETR")]
BLUE, TEAL, ORANGE, GRAY = "#2563a6", "#087e8b", "#b65d28", "#798997"


def main():
    sources = {}

    def read(relative):
        raw = (ROOT / relative).read_bytes()
        sources[relative] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    replication = read("docs/experiments/outer_combo_20261012/replication_results_20261013.json")
    capped = read("reports/capped_winner_31976_20261013/final_results_20261013.json")
    comparison = read("reports/capped_winner_31976_20261013/final_comparison_20261013.json")
    previous = read("reports/capped_replication_31976_20261007/final_summary_20261010.json")
    references = read("experiments/capped_winner_31976_20261013/source/paired_references.json")
    plan = read("reports/capped_winner_31976_20261013/plan_summary.json")
    protocol = read("experiments/capped_winner_31976_20261013/source/experiments/capped_winner_31976_20261013/swinunetr/protocol.json")
    splits = read("docs/experiments/outer_combo_20261012/replication_inputs.json")["splits"]
    for path, expected in comparison["provenance"].items():
        assert sources[path] == expected, path
    cells = {(f, s) for f in range(3) for s in range(3)}
    runs = replication["runs"]
    assert replication["completed"] == replication["planned"] == len(runs) == 9
    assert {(r["fold"], r["seed"]) for r in runs} == cells
    for fold, split in splits.items():
        assert len(split["train"]) == 10 and len(split["val"]) == 52
        assert not set(split["train"]) & set(split["val"])
        for other in splits:
            if other != fold:
                assert not set(split["val"]) & set(splits[other]["val"])
    assert capped["status"] == "complete" and len(capped["runs"]) == 36
    assert capped["budget_flops"] == previous["budget_flops"] == 31976450667090
    assert capped["budget_scope"] == "optimization_only" and not capped["calibration_in_cap"]
    old = {(r["model"], r["objective"], r["fold"], r["seed"]): r
           for r in previous["runs"] + previous["reused_runs"]}
    assert len(old) == 504
    for r in references:
        assert r["final_dice"] == old[(r["model"], r["objective"], r["fold"], r["seed"])]["final_dice"]

    def aggregate(values):
        values = [100 * x for x in values]
        return {"n": len(values), "mean_percent": st.mean(values), "sd_pp": st.stdev(values),
                "min_percent": min(values), "max_percent": max(values)}

    standard = aggregate([r["macro_dice"] for r in runs])
    assert abs(standard["mean_percent"] - replication["overall"]["mean_dice"] * 100) < 1e-10
    assert abs(standard["sd_pp"] - replication["overall"]["sd_dice"] * 100) < 1e-10
    folds = {str(f): aggregate([r["macro_dice"] for r in runs if r["fold"] == f]) for f in range(3)}
    models = []
    for model, label in MODELS:
        rows = sorted([r for r in capped["runs"] if r["model"] == model], key=lambda r: (r["fold"], r["seed"]))
        assert len(rows) == 9 and {(r["fold"], r["seed"]) for r in rows} == cells
        existing = next(r for r in comparison["models"] if r["model"] == model)
        objectives = {key[1] for key in old if key[0] == model}
        best = max(objectives, key=lambda objective: st.mean(old[(model, objective, f, s)]["final_dice"] for f, s in cells))
        assert best == existing["previous_best_objective"]
        paired = []
        for r in rows:
            assert r["optimization_gflops"] <= capped["budget_flops"] / 1e9
            assert abs(r["optimization_gflops"] + r["unused_optimization_gflops"] - capped["budget_flops"] / 1e9) < 1e-7
            record = {"fold": r["fold"], "seed": r["seed"], "new_dice_percent": 100 * r["final_dice"]}
            for name, objective in [("baseline", "dice"), ("original_bands", "dice_bands"), ("previous_best", best)]:
                reference = old[(model, objective, r["fold"], r["seed"])]["final_dice"]
                if name != "previous_best":
                    assert r[name + "_final_dice"] == reference
                record[name + "_dice_percent"] = 100 * reference
                record["gain_vs_" + name + "_pp"] = 100 * (r["final_dice"] - reference)
            paired.append(record)
        cohorts = {}
        for name, group in [("all9", paired), ("excluding_development_cell8", [r for r in paired if (r["fold"], r["seed"]) != (0, 0)])]:
            cohorts[name] = {"n": len(group), "new_mean_percent": st.mean(r["new_dice_percent"] for r in group),
                             "new_sd_pp": st.stdev(r["new_dice_percent"] for r in group)}
            for ref in ("baseline", "original_bands", "previous_best"):
                cohorts[name][ref + "_mean_percent"] = st.mean(r[ref + "_dice_percent"] for r in group)
                cohorts[name]["gain_vs_" + ref + "_pp"] = st.mean(r["gain_vs_" + ref + "_pp"] for r in group)
                cohorts[name]["wins_vs_" + ref] = sum(r["gain_vs_" + ref + "_pp"] > 0 for r in group)
        assert abs(cohorts["all9"]["new_mean_percent"] - existing["new_mean_percent"]) < 1e-10
        assert cohorts["all9"]["wins_vs_previous_best"] == existing["paired_wins_vs_previous_best"]
        models.append({"model": model, "label": label, "previous_best_objective": best,
                       "cohorts": cohorts, "paired_runs": paired,
                       "updates": rows[0]["updates"], "optimization_gflops": rows[0]["optimization_gflops"],
                       "calibration_gflops": rows[0]["calibration_gflops"],
                       "prefix_updates": plan["schedules"][model]["prefix_updates"],
                       "ramp_updates": plan["schedules"][model]["constraint_warmup_updates"]})
    summary = {"as_of": "2026-10-13", "standard_replication": standard, "by_fold": folds,
               "standard_runs": runs, "winner": replication["winner"],
               "frozen_objective": protocol["winner_objective"],
               "depth_gradient_ratio": protocol["winner_depth_gradient_ratio"],
               "models": models, "budget_flops": capped["budget_flops"],
               "calibration_in_cap": False, "source_sha256": sources}
    (HERE / "results_update_20261013.json").write_text(json.dumps(summary, indent=2) + "\n")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "text.color": "#172b3a"})

    def save(fig, name, note):
        fig.text(.045, .105, note, fontsize=11, color="#52616b", linespacing=1.5)
        fig.text(.045, .035, "Updated 13 October 2026 | Verified data and sources in results_update_20261013.json", fontsize=9, color=GRAY)
        for ext in ("png", "svg"):
            fig.savefig(FIG / f"{name}.{ext}", facecolor="white")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(13.33, 7.5), dpi=150)
    fig.subplots_adjust(left=.10, right=.95, top=.77, bottom=.24)
    fig.text(.045, .93, "The frozen loss repeats across three seeds and three folds", fontsize=22, weight="bold")
    fig.text(.045, .865, "SwinUNETR | 10 training / 52 validation cases per fold | Maximum 75 epochs | Selected checkpoints", fontsize=12)
    table_rows = []
    for f in range(3):
        scores = [next(r["macro_dice"] * 100 for r in runs if r["fold"] == f and r["seed"] == s) for s in range(3)]
        table_rows.append([f"Fold {f}", *[f"{v:.3f}%" for v in scores],
                           f"{folds[str(f)]['mean_percent']:.3f} ± {folds[str(f)]['sd_pp']:.3f}"])
    ax.axis("off")
    table = ax.table(cellText=table_rows, colLabels=["", "Seed 0", "Seed 1", "Seed 2", "Mean ± seed SD"],
                     colWidths=[.16, .19, .19, .19, .27], cellLoc="center", bbox=[0, .17, 1, .83])
    table.auto_set_font_size(False)
    table.set_fontsize(15)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_facecolor("#e4f1f2" if row else "#087e8b")
        if row == 0:
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        elif col == 4:
            cell.get_text().set_weight("bold")
    ax.text(.5, .01, f"All 9 runs: {standard['mean_percent']:.3f}% ± {standard['sd_pp']:.3f} pp", ha="center", fontsize=20, weight="bold", color=TEAL)
    save(fig, "10_best_constraint_replication", "Seed 2 is lower in every fold. Fold 0 informed recipe selection; repeated seeds reuse the same validation cases.\nThis table establishes repeatability of scores; it is not a paired gain table or an independent test-set estimate.")

    fig, ax = plt.subplots(figsize=(13.33, 7.5), dpi=150)
    fig.subplots_adjust(left=.23, right=.93, top=.75, bottom=.25)
    fig.text(.045, .93, "At equal compute, the largest gain transfers to 3D U-Net", fontsize=22, weight="bold")
    fig.text(.045, .865, "New loss minus each model's strongest previous objective | Final checkpoint | Three folds × three seeds", fontsize=12)
    for y, model in enumerate(models):
        for i, r in enumerate(model["paired_runs"]):
            ax.scatter(r["gain_vs_previous_best_pp"], y + (i - 4) * .043, s=48,
                       color=ORANGE if (r["fold"], r["seed"]) == (0, 0) else BLUE, alpha=.75)
        a = model["cohorts"]["all9"]
        ax.scatter(a["gain_vs_previous_best_pp"], y, color=TEAL, marker="D", s=120, zorder=4)
        ax.text(14.8, y, f"{a['gain_vs_previous_best_pp']:+.2f} pp\n{a['wins_vs_previous_best']}/9 wins", va="center", fontsize=13, weight="bold")
    ax.axvline(0, color=GRAY, linewidth=1)
    ax.set_yticks(range(4), [r["label"] for r in models])
    ax.invert_yaxis()
    ax.set_xlim(-8.5, 18.5)
    ax.set_xlabel("Paired change in validation Dice (percentage points)")
    ax.grid(axis="x", color="#e2e8f0")
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=10)
    save(fig, "11_best_constraint_equal_flops", "Dots: paired runs; orange: fold 0 / seed 0; diamonds: means. Cap: 31,976.45 GFLOPs; calibration is additional.\nExcluding fold 0 / seed 0: U-Net +5.95 pp; SwinUNETR −0.38; nnU-Net −0.67; UNETR −1.68.")
    print(json.dumps({"replication": standard, "capped_runs": len(capped["runs"]), "source_hashes_verified": len(comparison["provenance"])}, indent=2))
    return summary


if __name__ == "__main__":
    main()
