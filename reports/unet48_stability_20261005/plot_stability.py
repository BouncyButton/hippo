"""Plot preserved stability curves, including individually identified seeds."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
runs = json.loads((ROOT/"screen_history.json").read_text())
for name in ["resize_history.json", "linear_history.json", "replication_history.json"]:
    file = ROOT/name
    if file.exists():
        runs += json.loads(file.read_text())
colors = {"constant_bg":"#7b8794", "constant_fg":"#d97706", "constant_beta2_099":"#8b5cf6", "resize_bg":"#0284c7", "linear_bg":"#059669"}
labels = {"constant_bg":"Original head; fixed LR", "constant_fg":"Foreground Dice", "constant_beta2_099":"AdamW beta2=0.99", "resize_bg":"Final resize-convolution", "linear_bg":"Linear class head"}
fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharex=True, sharey=True)
for run in runs:
    curve = run["curve"]
    updates = [int(row["step"]) for row in curve]
    recipe = run["recipe"]
    for ax, metric in zip(axes, ["macro_hard_dice", "anterior_hard_dice", "posterior_hard_dice"]):
        ax.plot(updates, [100*float(row[metric]) for row in curve], color=colors[recipe],
                lw=2 if run["seed"]==0 else 1.2, linestyle=["-","--",":"][run["seed"]],
                label=labels[recipe]+f" (seed {run['seed']})")
for ax,title in zip(axes,["Macro foreground Dice","Anterior hippocampus","Posterior hippocampus"]):
    ax.set_title(title,fontsize=12)
    ax.set_xlabel("Training updates (10 updates = one patient pass)",fontsize=9)
    ax.set_ylim(0,100)
    ax.axhspan(70,75,color="#059669",alpha=.08)
    ax.grid(alpha=.18)
    ax.spines[["top","right"]].set_visible(False)
axes[0].set_ylabel("Validation hard Dice (%)")
handles, legend = axes[0].get_legend_handles_labels()
fig.legend(handles,legend,loc="lower center",bbox_to_anchor=(.5,-.06),ncol=3,frameon=False,fontsize=9)
fig.suptitle("Width-48 residual U-Net: isolated interventions on the same ten training patients",fontsize=13)
fig.tight_layout(rect=[0,.12,1,.96])
fig.savefig(ROOT/"stability_curves.png",dpi=170,bbox_inches="tight")
fig.savefig(ROOT/"stability_curves.pdf",bbox_inches="tight")
