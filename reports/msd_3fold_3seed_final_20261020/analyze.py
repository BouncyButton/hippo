"""Reproduce the final MSD analysis from immutable Drive reports; no training."""
from pathlib import Path
import csv
import hashlib
import html
import itertools
import json
import math
import os
import statistics as st
import zipfile

OUT = Path(__file__).resolve().parent
REPO = OUT.parent.parent
PACKAGE = REPO / "reports/msd_missing_324_second_account_20261018/package"
os.environ["MPLCONFIGDIR"] = str(OUT / ".matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODELS = ["nnunet", "swinunetr", "unetr_vit_b16", "basic_unet16"]
LABELS = {"nnunet": "nnUNet", "swinunetr": "SwinUNETR",
          "unetr_vit_b16": "UNETR", "basic_unet16": "3D U-Net"}
SEEDS, FOLDS = [0, 2, 20], [0, 1, 2]
CAP = 31_976_450_667_090


def read(path):
    return json.loads(path.read_text())


def arm_label(arm):
    if arm == "dice":
        return "Baseline"
    return " + ".join("SDF" if part == "sdf" else part.capitalize()
                      for part in arm.removeprefix("dice_").split("_"))


def save_csv(name, rows):
    with (OUT / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


report_path = next(OUT.glob("report______bd*.json"))
report = read(report_path)
accounting = read(next(OUT.glob("accounting_report*.json")))
rows = report["runs"]
arms = [a["arm"] for a in report["aggregate"] if a["model"] == MODELS[0]]
by_key = {(r["model"], r["arm"], r["fold"], r["seed"]): r for r in rows}
expected = set(itertools.product(MODELS, arms, FOLDS, SEEDS))
assert len(rows) == len(by_key) == 504 and set(by_key) == expected
assert report["completed"] == report["total"] == 504
assert len(arms) == 14 and report["seeds"] == SEEDS
assert report["new_completed"] == report["new_total"] == accounting["completed_results"] == 324
assert accounting["unfinished_charged_optimization_flops"] == 0
assert accounting["unfinished_charged_calibration_flops"] == 0

# Validate content-addressed report downloads, and every scientific update/FLOP schedule.
hashes = {}
for p in OUT.glob("*.json"):
    if "______" in p.name:
        hashes[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
        assert hashes[p.name] == p.stem.split("______")[1]
protocols = {}
for r in rows:
    p = PACKAGE / "seed20_bundle/protocols" / r["protocol"] / "protocol.json"
    protocols.setdefault(str(p), read(p))
    protocol = protocols[str(p)]
    assert protocol["augmentation"] == "none"
    assert protocol["folds"] == FOLDS and protocol["budget_flops"] == CAP
    schedule = protocol["models"][protocol["trainer_alias"]]["objectives"][r["arm"]]
    assert r["updates"] == schedule["max_steps"]
    assert r["training_flops"] == schedule["consumed_training_flops"] <= CAP
    assert r["calibration_flops"] == schedule["calibration_flops"]
    base = by_key[r["model"], "dice", r["fold"], r["seed"]]
    assert math.isclose(r["gain_pp"], r["dice"] - base["dice"], abs_tol=1e-10)
    assert all(math.isfinite(r[k]) for k in ("dice", "training_dice", "gain_pp"))
    assert 0 <= r["dice"] <= 100 and 0 <= r["training_dice"] <= 100
    r["train_validation_gap_pp"] = r["training_dice"] - r["dice"]
    r["cap_used_percent"] = 100 * r["training_flops"] / CAP
historical = read(PACKAGE / "historical_reference.json")
assert len(historical) == 180
for old in historical:
    current = by_key[old["model"], old["arm"], old["fold"], old["seed"]]
    assert all(current[k] == v for k, v in old.items())

groups, fold_rows, seed_rows = [], [], []
for m, a in itertools.product(MODELS, arms):
    g = [by_key[m, a, f, s] for f, s in itertools.product(FOLDS, SEEDS)]
    b = [by_key[m, "dice", f, s] for f, s in itertools.product(FOLDS, SEEDS)]
    fmeans = [st.mean(x["dice"] for x in g if x["fold"] == f) for f in FOLDS]
    fgains = [st.mean(x["gain_pp"] for x in g if x["fold"] == f) for f in FOLDS]
    row = dict(model=m, condition=a, n=9, validation_dice=st.mean(fmeans),
               sd_fold_means=st.stdev(fmeans), gain_pp=st.mean(fgains),
               sd_fold_gains=st.stdev(fgains),
               training_dice=st.mean(x["training_dice"] for x in g),
               train_validation_gap_pp=st.mean(x["train_validation_gap_pp"] for x in g),
               positive_pairs=sum(x["gain_pp"] > 0 for x in g),
               positive_fold_means=sum(x > 0 for x in fgains),
               minimum_pair_gain_pp=min(x["gain_pp"] for x in g),
               maximum_pair_gain_pp=max(x["gain_pp"] for x in g),
               minimum_updates=min(x["updates"] for x in g),
               maximum_updates=max(x["updates"] for x in g))
    for k in ("fn", "fp", "ap_swaps"):
        row[f"mean_{k}_voxels_per_validation_run"] = st.mean(x[k] for x in g)
        row[f"{k}_count_change_percent"] = 100 * (sum(x[k] for x in g) / sum(x[k] for x in b) - 1)
    groups.append(row)
    supplied = next(x for x in report["aggregate"] if x["model"] == m and x["arm"] == a)
    for k, v in [("mean_dice", row["validation_dice"]),
                 ("mean_gain_pp", row["gain_pp"]),
                 ("sd_across_folds", row["sd_fold_means"])]:
        assert math.isclose(supplied[k], v, abs_tol=1e-10)
    for key, values, target in [("fold", FOLDS, fold_rows), ("seed", SEEDS, seed_rows)]:
        for v in values:
            subset = [x for x in g if x[key] == v]
            target.append(dict(model=m, condition=a, **{key: v},
                               validation_dice=st.mean(x["dice"] for x in subset),
                               gain_pp=st.mean(x["gain_pp"] for x in subset),
                               training_dice=st.mean(x["training_dice"] for x in subset)))

group_map = {(r["model"], r["condition"]): r for r in groups}
best = {m: max((r for r in groups if r["model"] == m), key=lambda x: x["validation_dice"])["condition"]
        for m in MODELS}
edge_rows = []
for m in MODELS:
    deltas = [by_key[m, "dice_bands_edge", f, s]["dice"] - by_key[m, "dice_bands", f, s]["dice"]
              for f, s in itertools.product(FOLDS, SEEDS)]
    edge_rows.append(dict(model=m, contrast="bands+edge minus bands", mean_delta_pp=st.mean(deltas),
                          positive_pairs=sum(d > 0 for d in deltas), minimum_delta_pp=min(deltas),
                          maximum_delta_pp=max(deltas)))

# One final archive is also independently verified as an audit spot check.
archive_path = next(OUT.glob("archive__*.zip"))
assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == archive_path.stem.split("__")[-1]
with zipfile.ZipFile(archive_path) as z:
    ack, cleanup, config = [json.loads(z.read(n)) for n in
                            ("archive_acknowledgment.json", "cleanup.json", "config.json")]
    completion = json.loads(z.read("completion.json"))
    assert ack["status"] == "verified" and cleanup["status"] == "complete"
    assert completion["status"] == "complete" and config["augmentation"] == "none"
    for field, filename in [("metrics_sha256", "metrics.csv"), ("config_sha256", "config.json"),
                            ("calibration_sha256", "calibration.json"),
                            ("validation_best_sha256", "validation_best.json")]:
        assert hashlib.sha256(z.read(filename)).hexdigest() == completion[field]
    v = json.loads(z.read("validation_latest.json"))
    spot = by_key["basic_unet16", "dice_bands_edge_sdf_depth", 2, 20]
    assert abs(spot["dice"] - 100*v["summary"]["macro_hard_dice"]) < 1e-10

audit = dict(completed=504, unique_grid_cells=504, new_completed=324, preserved_historical=180,
             folds=FOLDS, seeds=SEEDS, augmentation="none", cap_flops_per_trajectory=CAP,
             protocol_schedule_mismatches=0, recalculated_aggregate_mismatches=0,
             unfinished_charged_flops=0, report_sha256=hashes,
             completed_optimization_flops=sum(r["training_flops"] for r in rows),
             completed_calibration_flops=sum(r["calibration_flops"] for r in rows),
             recovery_optimization_flops_lower_bound=accounting["recovery_overhead_optimization_flops_lower_bound"],
             recovery_calibration_flops_lower_bound=accounting["recovery_overhead_calibration_flops_lower_bound"],
             incomplete_recovery_accounting_tasks=accounting["overhead_not_fully_known_for"],
             archive_spot_check=archive_path.name, archive_spot_check_cleanup="complete",
             scope="All summary rows, saved protocol schedules and report hashes checked. "
                   "The 180 historical summary rows exactly match the fixed prior audit. "
                   "One final raw archive independently checked; this is not a new download/hash audit of all 504 raw archives.")
(OUT / "analysis_audit.json").write_text(json.dumps(audit, indent=2))
save_csv("all_504_runs.csv", rows)
save_csv("all_56_condition_summaries.csv", groups)
save_csv("fold_means_and_gains.csv", fold_rows)
save_csv("seed_means_and_gains.csv", seed_rows)
save_csv("edge_added_to_bands.csv", edge_rows)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.facecolor": "white"})
fig, ax = plt.subplots(figsize=(9.6, 8.0), layout="constrained")
values = np.array([[group_map[m, a]["gain_pp"] for m in MODELS] for a in arms[1:]])
im = ax.imshow(values, cmap="RdBu", vmin=-12, vmax=12, aspect="auto")
ax.set_xticks(range(4), [LABELS[m] for m in MODELS])
ax.set_yticks(range(13), [arm_label(a) for a in arms[1:]])
ax.tick_params(length=0)
for i, j in itertools.product(range(13), range(4)):
    ax.text(j, i, f"{values[i,j]:+.2f}", ha="center", va="center",
            color="white" if abs(values[i,j]) > 7.5 else "#101828")
fig.colorbar(im, ax=ax, label="Validation Dice gain over matched baseline (percentage points)", shrink=.8)
ax.set_title("MSD: constraint gains across folds 0/1/2 and seeds 0/2/20\n"
             "504 runs · no augmentation · final budget checkpoint", loc="left", pad=16, fontweight="bold")
fig.savefig(OUT/"constraint_gains.png", dpi=180)
fig.savefig(OUT/"constraint_gains.pdf")
plt.close(fig)

fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.2), layout="constrained")
colors = ["#6b7280", "#0072B2", "#D55E00", "#009E73"]
for ax, m in zip(axes.flat, MODELS):
    chosen = ["dice", "dice_bands", "dice_bands_edge", best[m]]
    for i, a in enumerate(chosen):
        g = [by_key[m, a, f, s] for f, s in itertools.product(FOLDS, SEEDS)]
        ax.scatter(np.full(9, i)+np.tile([-.10,0,.10],3), [r["dice"] for r in g],
                   color=colors[i], alpha=.38, s=24)
        means = [st.mean(x["dice"] for x in g if x["fold"] == f) for f in FOLDS]
        ax.scatter(np.full(3,i)+[-.15,0,.15],means,marker="D",s=44,color=colors[i],edgecolor="white",zorder=3)
        ax.plot([i-.26,i+.26],[st.mean(means)]*2,color="#101828",lw=2)
    ax.set_title(LABELS[m], fontweight="bold")
    ax.set_xticks(range(4), ["Baseline", "Bands", "Bands+edge", "Best mean"])
    ax.set_ylabel("Validation Dice (%)")
    ax.set_ylim(20,90)
    ax.grid(axis="y",alpha=.18)
fig.suptitle("Variation across the nine runs\nFaint circles: individual runs · diamonds: fold means · bars: overall means",
             fontsize=13, fontweight="bold")
fig.savefig(OUT/"replicate_variation.png",dpi=170)
fig.savefig(OUT/"replicate_variation.pdf")
plt.close(fig)

fig, axes = plt.subplots(1,4,figsize=(12,3.8),layout="constrained",sharey=True)
for ax,m in zip(axes,MODELS):
    for i,a in enumerate(["dice","dice_bands","dice_bands_edge",best[m]]):
        g=group_map[m,a]
        ax.plot([0,1],[g["training_dice"],g["validation_dice"]],marker="o",
                color=colors[i],label=["Baseline","Bands","Bands+edge","Best mean"][i])
    ax.set_title(LABELS[m],fontweight="bold")
    ax.set_xticks([0,1],["Training","Validation"])
    ax.grid(axis="y",alpha=.18)
axes[0].set_ylabel("Final Dice (%)")
axes[-1].legend(fontsize=8,loc="lower left")
fig.suptitle("Final training versus validation Dice",fontweight="bold")
fig.savefig(OUT/"training_validation.png",dpi=170)
plt.close(fig)


def table(headers, body):
    return "<div class='table-wrap'><table><thead><tr>"+ "".join("<th>"+html.escape(str(h))+"</th>" for h in headers)+"</tr></thead><tbody>"+ "".join("<tr>"+"".join("<td>"+str(c)+"</td>" for c in r)+"</tr>" for r in body)+"</tbody></table></div>"


matrix = table(["Added constraints"]+[LABELS[m] for m in MODELS],
               [[arm_label(a)]+[f'{group_map[m,a]["validation_dice"]:.2f}<br><small>{group_map[m,a]["gain_pp"]:+.2f} pp</small>' for m in MODELS] for a in arms])
chosen_rows = []
for m in MODELS:
    b=group_map[m,"dice"];w=group_map[m,best[m]]
    chosen_rows.append([LABELS[m],f'{b["validation_dice"]:.2f}',arm_label(best[m]),
                        f'{w["validation_dice"]:.2f} ± {w["sd_fold_means"]:.2f}',
                        f'{w["gain_pp"]:+.2f}',f'{w["positive_pairs"]}/9'])
winner_table=table(["Model","Baseline","Best observed mean","Dice ± fold SD","Gain (pp)","Positive pairs"],chosen_rows)
seed_table=table(["Model","Condition"]+[f"Seed {s}" for s in SEEDS],
                 [[LABELS[m],arm_label(a)]+[f'{next(r["validation_dice"] for r in seed_rows if r["model"]==m and r["condition"]==a and r["seed"]==s):.2f}' for s in SEEDS]
                  for m in MODELS for a in ["dice",best[m]]])
err_table=table(["Model","Condition","FN voxels","FP voxels","A/P swaps","FN change","FP change","A/P change"],
                [[LABELS[m],arm_label(a)]+[f'{group_map[m,a][f"mean_{k}_voxels_per_validation_run"]:,.0f}' for k in ("fn","fp","ap_swaps")]+
                 [f'{group_map[m,a][f"{k}_count_change_percent"]:+.1f}%' for k in ("fn","fp","ap_swaps")]
                 for m in MODELS for a in ["dice","dice_bands",best[m]]])
train_table=table(["Model","Condition","Training Dice","Validation Dice","Gap (pp)"],
                  [[LABELS[m],arm_label(a)]+[f'{group_map[m,a][k]:.2f}' for k in ("training_dice","validation_dice","train_validation_gap_pp")]
                   for m in MODELS for a in ["dice","dice_bands",best[m]]])
edge_table=table(["Model","Bands+edge minus bands (pp)","Positive pairs"],
                 [[LABELS[r["model"]],f'{r["mean_delta_pp"]:+.3f}',f'{r["positive_pairs"]}/9'] for r in edge_rows])
budget_table=table(["Model","Updates per run","Optimization cap used"],
                   [[LABELS[m],f'{min(r["updates"] for r in rows if r["model"]==m)}–{max(r["updates"] for r in rows if r["model"]==m)}',
                     f'{min(r["cap_used_percent"] for r in rows if r["model"]==m):.2f}–{max(r["cap_used_percent"] for r in rows if r["model"]==m):.2f}%'] for m in MODELS])
html_report = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<title>MSD — final 504-run analysis</title><style>
body{{font:16px/1.6 system-ui,sans-serif;color:#182332;background:#f5f7fa;margin:0}}
main{{max-width:1140px;margin:auto;padding:48px 32px;background:white}}
h1{{font-size:38px;line-height:1.15}}h2{{margin-top:44px;font-size:24px}}.eyebrow{{letter-spacing:.12em;font-size:12px;color:#526777;font-weight:700}}
.lead{{font-size:20px;max-width:960px}}.note{{padding:18px 22px;background:#eef4fa;border-left:4px solid #0072B2}}small{{color:#5b6574}}
table{{border-collapse:collapse;width:100%;font-size:14px}}th{{background:#edf2f7;text-align:left}}td,th{{padding:10px 12px;border-bottom:1px solid #dee5ed;vertical-align:top}}
.table-wrap{{overflow-x:auto}}img{{width:100%;height:auto}}a{{color:#0067a0}}li{{margin:.6em 0}}
@media print{{main{{padding:15px}}h2{{break-after:avoid}}table,img{{break-inside:avoid}}body{{background:white}}}}
</style><main><div class="eyebrow">MSD · FINAL VALIDATION ANALYSIS · 20 OCTOBER 2026</div>
<h1>Bands help every model.<br>The best combination depends on the architecture.</h1>
<p class="lead">All 504 planned cells are present: four models × 14 conditions × folds 0/1/2 × seeds 0/2/20.
Bands beat each matched baseline in all 36 model/fold/seed comparisons. The highest overall mean is
3D U-Net with bands+edge+depth: 80.00% Dice, up 6.17 percentage points.</p>
<p class="note"><b>Interpretation:</b> This is an exploratory, post-selection analysis. Seed 20 replaced seed 1 after results were inspected.
The old seed-1 results remain separate. These are validation results, not an independent test set; selecting the best of 14
conditions on the same validation data also favors the reported winners.</p>
<h2>Main comparison</h2>{winner_table}
<p>“Best” means the highest observed mean, not proven superiority over nearby combinations.
The gap to the runner-up is only 0.17 pp for nnUNet, 0.30 pp for SwinUNETR, 0.21 pp for UNETR,
and 0.10 pp for 3D U-Net. Each selected winner improves all nine matched baseline pairs.</p>
<h2>All 14 conditions</h2>
<p>Each cell shows final validation Dice (%) and the change from the same model’s baseline.
Average three seeds within each fold, then average the three folds equally. Reported SD is the sample
SD of three fold means; it is not a confidence interval and does not describe seed variability.</p>
{matrix}<img src="constraint_gains.png" alt="Heatmap of all matched constraint gains">
<h2>What the constraints are doing</h2><ul>
<li><b>nnUNet:</b> bands provide +7.55 pp. Bands+SDF+depth reaches +10.15 pp. Adding edge to bands reduces
Dice in every pair (−1.09 pp on average). SDF alone reduces Dice by 2.04 pp.</li>
<li><b>SwinUNETR:</b> depth is the strongest single tested addition (+8.24 pp); bands add +3.85 pp.
The full combination reaches +11.97 pp, although much of that gain comes from the weak seed-0 baseline.</li>
<li><b>UNETR:</b> bands help in all nine pairs (+3.89 pp). Bands+edge+SDF reaches +5.35 pp.
Adding depth to that combination lowers the mean by 0.47 pp, so more constraints are not automatically better.</li>
<li><b>3D U-Net:</b> bands add +4.83 pp and bands+edge adds +5.71 pp. Bands+edge+depth reaches +6.17 pp.
The full combination is essentially tied at −0.10 pp relative to that best mean.</li></ul>
<p>The edge effect is small and model dependent:</p>{edge_table}
<h2>Seed and fold consistency</h2>
<p>Bands and each model’s best-mean condition improve every fold mean and every individual pair.
However, averaging seeds hides substantial initialization variability, especially for nnUNet and SwinUNETR.</p>
{seed_table}
<p>Seed-2 nnUNet remains weaker (48.96% baseline, 62.98% with the best-mean condition).
Swin’s best-mean gain is +25.93 pp for seed 0, +1.64 pp for seed 2, and +8.32 pp for seed 20.
By contrast, 3D U-Net’s best condition averages 80.32%, 79.46%, and 80.21% across seeds.
Keep all three selected seeds in the reported analysis; these differences are evidence about robustness.</p>
<img src="replicate_variation.png" alt="Individual results and fold means for baseline, bands, bands plus edge, and best condition">
<h2>Training versus validation</h2>{train_table}
<img src="training_validation.png" alt="Final training and validation Dice by model and condition">
<p>The improvements are present on validation data. nnUNet’s gap stays close to 4 pp.
UNETR’s gap grows from 7.36 to 10.25 pp, and 3D U-Net’s from 4.67 to 7.66 pp at their best conditions:
the constraints improve validation performance but do not eliminate the generalization gap.
These are final endpoint comparisons, not longitudinal training curves.</p>
<h2>False negatives, false positives and anterior/posterior swaps</h2>
<p>Counts below are average voxel totals per 52-case validation run. Percent changes compare summed counts
across the same nine matched runs; they are <b>not</b> false-positive/false-negative rates.
Foreground FN means hippocampus predicted as background; FP means background predicted as hippocampus.
Anterior/posterior confusion is counted separately.</p>{err_table}
<p>nnUNet’s improvement primarily reduces missed foreground (−33.5% at its best mean), with FP increasing 4.5%.
SwinUNETR, UNETR and 3D U-Net chiefly reduce excess foreground predictions (FP −59.9%, −23.8%, −42.2%),
at the cost of increased FN. Dice improves because this tradeoff is favorable overall.
The large relative FN increases for Swin start from a very small baseline count.
A/P swaps improve much less than FP, so the results do not establish a dramatic improvement in anatomical subregion separation.</p>
<h2>Scientific settings and compute audit</h2>
<p>All rows match their frozen protocol’s exact update count, optimization FLOPs and calibration FLOPs.
All protocols specify <b>no augmentation</b>, ten training volumes and 52 validation volumes per fold,
64³ inputs and batch size one. The primary metric uses the final budget checkpoint, not the best validation checkpoint.</p>
{budget_table}
<p>The optimization cap is 31,976,450,667,090 FLOPs (31.98 TFLOPs) per trajectory.
Completed trajectories total 16.0548 PFLOPs of optimization plus 0.2732 PFLOPs of separately charged calibration.
The recovery report for the 324-run continuation additionally records at least 0.2073 PFLOPs of optimization overhead
and 0.0036 PFLOPs of calibration overhead. Seven old truncated ledgers prevent exact recovery-cost totals.
There are zero unfinished compute charges. A completed trajectory’s cap does not include recovery/replay or calibration.</p>
<p>nnUNet uses native foreground Dice+CE, deep supervision and SGD with 60% auxiliary-gradient targets
and a 75% p95 ceiling. The other three use MONAI Dice including background and AdamW, with 10% targets
and a 50% p95 ceiling. Architecture, baseline loss, optimizer, and constraint strength are therefore confounded:
the within-model comparisons are stronger evidence than a claim that one architecture is intrinsically superior.</p>
<h2>Verification and provenance</h2>
<p>The signed-content hashes of all downloaded reports match their filenames. The combined report contains 504 unique,
complete grid cells; all 56 aggregate rows were independently recalculated. The 180 reused summaries exactly match
the fixed prior audit, and the continuation contributes all 324 missing results. One final raw archive was independently
checked against its hashes, saved configuration and metric files; its server archive acknowledgment is verified and its
local generated-checkpoint cleanup is complete. This analysis does not claim a fresh re-download of all 504 raw archives.</p>
<p><a href="https://drive.google.com/file/d/1f0-S6phDP7LuTHCprp50rmlhKOWDfzG4/view">Final combined report on Drive</a> ·
<a href="https://drive.google.com/file/d/1f_atLAmltRJ7Gpx-fumJtBtqmHzbvPkj/view">Recovery accounting on Drive</a> ·
<a href="https://drive.google.com/drive/folders/1ABBWxmuJIB-7CvoAF6iPe3bz75bI3zuL">Original archives</a></p>
<h2>Download the analysis</h2><ul>
<li><a href="all_56_condition_summaries.csv">All 56 condition summaries, error counts and paired wins</a></li>
<li><a href="all_504_runs.csv">All 504 individual runs and matched gains</a></li>
<li><a href="fold_means_and_gains.csv">Fold means and gains</a> · <a href="seed_means_and_gains.csv">Seed means and gains</a></li>
<li><a href="analysis_audit.json">Machine-readable audit</a> · <a href="analyze.py">Reproducible analysis script</a></li>
<li><a href="constraint_gains.pdf">Gain heatmap PDF</a> · <a href="replicate_variation.pdf">Variation plot PDF</a></li></ul>
</main></html>"""
(OUT / "analysis.html").write_text(html_report)
print(json.dumps(dict(audit=audit, best=best, bands_edge_increment=edge_rows), indent=2))
