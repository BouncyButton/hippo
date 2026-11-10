"""Rebuild aggregate-only meeting figures from preserved experiment evidence."""
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
FIG.mkdir(exist_ok=True)
sources = {}


def read(rel):
    path = ROOT / rel
    data = path.read_bytes()
    sources[rel] = hashlib.sha256(data).hexdigest()
    return json.loads(data)


flops = read("reports/capped_replication_31976_20261007/final_summary_20261010.json")
assert flops["status"] == "complete" and len(flops["runs"]) == 448
assert len({(r["model"], r["objective"], r["fold"], r["seed"]) for r in flops["runs"]}) == 448
models = [("nnunet", "nnU-Net"), ("swinunetr", "SwinUNETR"),
          ("unetr_vit_b16", "UNETR"), ("basic_unet16", "3D U-Net")]
compute = []
for model, label in models:
    rows = [r for r in flops["runs"] if r["model"] == model]
    assert len(rows) == 112
    base = [r for r in rows if r["objective"] == "dice"]
    bands = [r for r in rows if r["objective"] == "dice_bands"]
    assert len(base) == len(bands) == 8
    objectives = {r["objective"] for r in rows}
    best = max(objectives, key=lambda o: st.mean(r["final_dice"] for r in rows if r["objective"] == o))
    selected = [r for r in rows if r["objective"] == best]
    compute.append({"model": label, "baseline": st.mean(r["final_dice"] for r in base)*100,
                    "bands": st.mean(r["final_dice"] for r in bands)*100,
                    "bands_gains": [r["gain_vs_paired_baseline_pp"] for r in bands],
                    "best_objective": best, "best": st.mean(r["final_dice"] for r in selected)*100,
                    "best_gain": st.mean(r["gain_vs_paired_baseline_pp"] for r in selected),
                    "baseline_updates": sorted({r["updates"] for r in base})})

geo = read("reports/geofwi3d_replication_20261011/results/summary.json")
assert geo["state"] == "complete" and len(geo["runs"]) == 54
schedule = read("docs/experiments/low_data_independent_audit_20260919/AUDIT.json")["aggregate"]
branch = read("docs/experiments/native_branch_constraints_20261008/results.json")
assert branch["completed"] == branch["planned"] == 12
branch_rows = {r["objective"]: r for r in branch["rows"]}
band = read("docs/thesis/supervisor_meeting_20261012/band_status_snapshot_update.json")
winner = band["top_rows"][0]
latest_replication = read("docs/experiments/outer_combo_20261012/replication_results_20261013.json")
controls = {r["objective"]: r for r in band["controls"]}
progression = read("docs/experiments/band_focus_20261011_recovery/progression_records_20261012.json")
winner_plan = read("docs/experiments/band_focus_20261011_recovery/winner_plan.json")
planned = {r["id"]: r for r in winner_plan["new_runs"]}
assert planned[winner["objective"]] == band["winner_details"]["config.json"]["selected"]["objective"]
records = progression["records"]
for row in band["all_rows"]:
    value = 100*float(row["macro_dice"])
    if value <= records[-1]["dice_percent"] + 1e-10:
        continue
    spec = planned[row["objective"]]["band"]
    records.append({"step":len(records)+1, "objective":row["objective"],
                    "description":f"3/2 mm bands; hardest {spec['inside_fraction']:.0%} inside / {spec['outside_fraction']:.0%} outside; emphasis {spec['inside_hard_mix']:.0%} inside / {spec['outside_hard_mix']:.0%} outside",
                    "dice_percent":value,"gain_pp":value-records[-1]["dice_percent"],
                    "campaign":"Winner refinement", "best_epoch":int(row["best_epoch"]),
                    "source":"band_status_snapshot_update.json"})
assert records[-1]["objective"] == winner["objective"]
progression.update(retrieved_at_utc=band["retrieved_at_utc"],
                   gain_over_dice_pp=records[-1]["dice_percent"]-records[0]["dice_percent"],
                   gain_over_original_bands_pp=records[-1]["dice_percent"]-records[1]["dice_percent"],
                   gap_to_branch_pp=progression["branch_reference"]["dice_percent"]-records[-1]["dice_percent"])
(HERE/"progression_records_updated.json").write_text(json.dumps(progression,indent=2)+"\n")

# The preserved report has three-decimal percentages. Do not invent extra precision.
archive = ROOT / "reports/cluster_storage_20261003/evidence.jsonl"
medsam_report = None
for line in archive.open():
    row = json.loads(line)
    if row.get("path") == "medsam3_multiclass_replication_20260930/analysis/RESULTS.md":
        medsam_report = row["text"]
assert medsam_report is not None
assert hashlib.sha256(medsam_report.encode()).hexdigest() == "0e01c1a3b4a99cf0db645f24fde5d13ffff5059692e5b523a85831653ead70d1"
(HERE / "medsam3_replication_preserved_report.md").write_text(medsam_report)
sources["reports/cluster_storage_20261003/evidence.jsonl#medsam3_multiclass_replication_20260930/analysis/RESULTS.md"] = hashlib.sha256(medsam_report.encode()).hexdigest()
medsam = []
for line in medsam_report.splitlines():
    cells = [x.strip() for x in line.strip().strip("|").split("|")]
    if line.startswith("|") and len(cells) == 11 and cells[0] in {"0", "1", "3"}:
        medsam.append({"fold": int(cells[0]), "arm": cells[1], "dice": float(cells[2]),
                       "gain_pp": float(cells[3]), "seed_sd_pp": float(cells[4]),
                       "positive_seeds": cells[10]})
assert len(medsam) == 9
medsam_overall = {arm: st.mean(r["dice"] for r in medsam if r["arm"] == arm)
                  for arm in ["baseline", "bands", "bands_edge"]}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 14,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.spines.left": False, "axes.titleweight": "bold",
                     "axes.labelcolor": "#334155", "text.color": "#172b3a",
                     "xtick.color": "#475569", "ytick.color": "#334155"})
BLUE, TEAL, ORANGE, GRAY = "#2563a6", "#087e8b", "#b65d28", "#798997"


def canvas(title, subtitle, left=.22):
    fig, ax = plt.subplots(figsize=(13.33, 7.5), dpi=150)
    fig.subplots_adjust(left=left, right=.90, top=.75, bottom=.24)
    fig.text(.045, .93, title, fontsize=24, weight="bold")
    fig.text(.045, .865, subtitle, fontsize=13, color="#52616b")
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#e2e8f0", linewidth=.8)
    ax.tick_params(axis="y", length=0, pad=12)
    return fig, ax


def save(fig, name, note, date="13 October 2026"):
    fig.text(.045, .095, note, fontsize=11, color="#52616b", linespacing=1.5)
    fig.text(.045, .035, f"Thesis supervisor meeting | {date} | Values and sources in BRIEFING.md", fontsize=9, color="#798997")
    for ext in ["png", "svg"]:
        fig.savefig(FIG / f"{name}.{ext}", facecolor="white")
    plt.close(fig)


fig, ax = canvas("Bands improve mean Dice under the FLOP cap", "Eight new fold/seed combinations, excluding the selection pilot | Final checkpoint | MSD")
for y, row in enumerate(compute):
    gains = row["bands_gains"]
    ax.scatter(gains, [y+(i-3.5)*.045 for i in range(8)], color=BLUE, alpha=.38, s=40)
    mean = st.mean(gains)
    ax.scatter([mean], [y], color=TEAL, marker="D", s=110, zorder=4)
    ax.text(21, y, f"+{mean:.2f} pp", va="center", fontsize=16, weight="bold")
ax.set_yticks(range(4), [r["model"] for r in compute]); ax.invert_yaxis()
ax.axvline(0, color=GRAY, linewidth=1); ax.set_xlim(-5, 24)
ax.set_xlabel("Paired Dice change from adding bands (percentage points)")
save(fig, "01_compute_bands", "Small circles: individual paired gains. Diamonds: means. UNETR improves in 5/8 combinations; the other models in 8/8.\nCap: 31,976.45 GFLOPs per run. Calibration is additional; these are exploratory validation results.")

fig, ax = canvas("MedSAM3 gains repeat across folds and seeds", "Three-class study | Five support volumes per fold | 200 updates | Three seeds per fold")
labels = {"baseline":"Baseline", "bands":"+ bands", "bands_edge":"+ bands + edge"}
colors = {"baseline": GRAY, "bands":BLUE, "bands_edge":TEAL}
for y, fold in enumerate([0, 1, 3]):
    rows = [r for r in medsam if r["fold"] == fold]
    ax.plot([r["dice"] for r in rows], [y]*3, color="#cbd5e1", linewidth=2, zorder=1)
    for r in rows:
        ax.scatter(r["dice"], y, s=100, color=colors[r["arm"]], label=labels[r["arm"]] if y==0 else None)
        ax.annotate(f'{r["dice"]:.2f}', (r["dice"],y), xytext=(0,-25 if r["arm"] == "bands_edge" else 14), textcoords="offset points", ha="center", fontsize=12, color=colors[r["arm"]])
ax.set_yticks(range(3), ["Fold 0", "Fold 1", "Fold 3"]); ax.invert_yaxis(); ax.set_ylim(2.5,-.55)
ax.set_xlim(71,84); ax.set_xlabel("Mean anterior/posterior validation Dice (%)")
handles, legend_labels = ax.get_legend_handles_labels()
fig.legend(handles, legend_labels, loc="center", bbox_to_anchor=(.55,.80), ncol=3, frameon=False, fontsize=12)
save(fig, "02_medsam3", "Equal-fold means: 74.93% baseline, 78.76% bands, 80.12% bands + edge. Bands + edge improves 9/9 comparisons.\nSelected folds were informed by earlier binary results. The seed-46 pilot is separate. Zoomed score axis.")

fig, ax = canvas("The constraint benefit extends to GeoFWI3D", "SwinUNETR | 19 training volumes per fold | 500 updates | Three folds × three seeds", left=.29)
methods=[("bands","Bands"),("sdf","Signed distance"),("bands_sdf","Bands + SDF"),("bands_edge","Bands + edge"),("bands_sdf_edge","Bands + SDF + edge")]
for y,(method,label) in enumerate(methods):
    gains=[r["gain_vs_baseline_pp"] for r in geo["runs"] if r["method"]==method]
    ax.scatter(gains,[y+(i-4)*.035 for i in range(9)],color=BLUE,alpha=.4,s=35)
    m=geo["overall"][method]["mean_paired_gain_pp"]
    ax.scatter(m,y,color=TEAL,s=90,marker="D",zorder=4)
    ax.text(5.6,y,f"+{m:.2f} pp",va="center",weight="bold",fontsize=15)
ax.axvline(0,color=GRAY,linewidth=1);ax.set_yticks(range(5),[x[1] for x in methods]);ax.invert_yaxis()
ax.set_xlim(-2,6.7);ax.set_xlabel("Paired foreground Dice change (percentage points)")
save(fig,"03_geofwi3d","Baseline mean: 71.53%. Bands + SDF + edge: 73.54%, improving all 9 comparisons. Six pilot results are included.\nBest validation checkpoint within 500 updates; equal updates do not imply equal FLOPs. Synthetic salt segmentation.")

fig, ax = canvas("Improved bands approach the auxiliary branch", "MSD | Fold 0, seed 0 | 10 training / 52 validation volumes | Validation-selected checkpoints",left=.34)
names=["Plain SwinUNETR / Dice", "Original bands", "Focused bands + endpoints + depth", "Native branch / Dice", "Native branch / bands + edge"]
vals=[100*float(controls["dice"]["macro_dice"]),100*float(controls["bands"]["macro_dice"]),100*float(latest_replication["winner"]["macro_dice"]),100*branch_rows["dice"]["macro_dice"],100*branch_rows["dice_bands_edge"]["macro_dice"]]
for y,(name,val) in enumerate(zip(names,vals)):
    color=ORANGE if y>2 else (TEAL if y==2 else BLUE)
    ax.scatter(val,y,s=125,color=color)
    ax.text(val+.22,y,f"{val:.2f}%",va="center",color=color,fontsize=16,weight="bold")
ax.set_yticks(range(5),names);ax.invert_yaxis();ax.set_xlim(72,86);ax.set_xlabel("Mean anterior/posterior validation Dice (%)")
save(fig,"04_bands_branch",f"Frozen recipe: +{vals[2]-vals[1]:.2f} pp vs original bands, {vals[4]-vals[2]:.2f} pp below the strongest recorded native-branch arm.\nDevelopment-cell scores only; the separate nine-run loss mean is 80.22%. Architectures and compute differ. Zoomed axis.")

fig, ax = plt.subplots(figsize=(13.33,7.5),dpi=150)
fig.subplots_adjust(left=.09,right=.965,top=.80,bottom=.23)
fig.text(.045,.94,"Bands progression: development snapshot at 81.112%",fontsize=22,weight="bold")
fig.text(.045,.875,"Ordinary SwinUNETR | 10 training / 52 validation volumes | Fold 0, seed 0 | Maximum 75 epochs",fontsize=12,color="#52616b")
ax.spines['left'].set_visible(True)
ax.grid(axis='y',color='#e2e8f0');ax.set_axisbelow(True)
xs=[r['step'] for r in records];ys=[r['dice_percent'] for r in records]
ax.plot(xs,ys,'o-',color=TEAL,linewidth=2.4,markersize=4.5)
reference=progression['branch_reference']['dice_percent']
ax.axhline(reference,color=ORANGE,linestyle='--',linewidth=1.6)
ax.text(1.1,reference+.07,f'Native auxiliary branch + bands + edge: {reference:.3f}%',color=ORANGE,fontsize=12)
annotations=[(1,'Dice only\n74.304%',(1.4,75.1),'left'),
             (2,'Original bands\n76.727%',(3.2,75.9),'left'),
             (8,'Separate 1+3 mm targets\n78.132%',(6.7,78.65),'center'),
             (11,'Hard-voxel focus\n78.937%',(11,77.55),'center'),
             (15,'3/2 mm bands\n79.263%',(15.8,78.45),'center'),
             (21,'Previous best\n80.725%',(19.2,81.12),'center'),
             (len(records),'70% / 10% selection\n65% / 85% emphasis\n81.112%',(24.6,79.40),'right')]
for step,label,xytext,ha in annotations:
    ax.annotate(label,(step,ys[step-1]),xytext=xytext,ha=ha,fontsize=10.5,
                arrowprops=dict(arrowstyle='-',color=GRAY,lw=.8))
ax.set_xlim(.5,len(records)+.8);ax.set_ylim(73.8,82.65)
ax.set_xticks([1,2,5,8,11,15,18,21,len(records)])
ax.set_ylabel('Validation A/P Dice (%)');ax.set_xlabel('Successive validation record (see numbered speaker notes)')
save(fig,'07_progression_updated',"Historical snapshot through 12 October at 15:10; subsequent selected recipe reaches 81.502%.\nCompleted variants only. Adaptive development on one reused split; record points are not cumulative loss additions.",date="12 October 2026 snapshot")

fig, ax = canvas("The bands advantage shrinks with longer schedules", "Historical matched SwinUNETR comparisons | Fold 0, three seeds | Validation-selected A/P Dice",left=.31)
keys=["fold0_75","fold0_uniform_200","fold0_adaptive_200_200_400"]
names=["Maximum 75 epochs", "Maximum 200 epochs", "Mixed 200 / 200 / 400"]
for y,key in enumerate(keys):
    val=schedule[key]["best_gain_pp"]
    ax.scatter(val,y,s=125,color=ORANGE if y==2 else TEAL)
    ax.text(val+.11,y,f"+{val:.2f} pp",va="center",fontsize=17,weight="bold")
ax.set_yticks(range(3),names);ax.invert_yaxis();ax.set_ylim(2.5,-.5);ax.set_xlim(0,5)
ax.set_xlabel("Bands minus baseline (percentage points)")
save(fig,"05_schedule_sensitivity","The common 200-epoch policy leaves +0.74 pp. The mixed schedule leaves +0.31 pp and includes an adaptive extension.\nDuration and learning-rate decay timing both change. These results do not prove disappearance or convergence.")

from build_results_update import main as build_results_update
new_results = build_results_update()
summary={"as_of":"2026-10-13","compute_replication8":compute,"medsam3_fold_means":medsam,"medsam3_equal_fold_means_from_rounded_report":medsam_overall,
         "geofwi3d_overall":geo["overall"],"historical_schedule_comparisons":schedule,
         "band_comparison":{"labels":["plain_dice","original_bands","improved_bands","native_branch_dice","native_branch_bands_edge"],"dice_percent":vals,
                            "gain_over_original_bands_pp":vals[2]-vals[1],"gap_to_branch_pp":vals[4]-vals[2]},
         "latest_standard_replication":new_results["standard_replication"],
         "latest_capped_comparison":new_results["models"],
         "source_sha256":{**sources,**new_results["source_sha256"]}}
(HERE/"verified_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps({"figures":len(list(FIG.glob('*.png'))),"medsam3_means":medsam_overall,"band_gap_pp":vals[4]-vals[2],"compute_new_runs":448,"geo_cells":54},indent=2))
