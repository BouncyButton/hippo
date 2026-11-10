"""Baseline/selected-objective chart with explicit constraint combinations.

Scores, fold SDs and paired gains retain the verified original chart data.
The twelve selected objectives are checked against the thesis's paired summary.
"""
from pathlib import Path
import hashlib
import json
import os
import re

os.environ.setdefault("MPLCONFIGDIR", "/tmp/thesis-dataset-methods-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "reports/thesis_results_readability_20261102/chart_data.json"
REPORT = ROOT / "reports/thesis_scratch_objective_labels_20261103"
OUTPUT = ROOT / "Template_tesi/figures/compiled/scratch-key-results.pdf"


def main():
    groups = json.loads(DATA.read_text())["scratch"]
    tex = (ROOT / "Template_tesi/thesis-revision.tex").read_text()
    table_end = tex.index(r"\label{tab:paired-replication-summary}")
    table_start = tex.rfind(r"\begin{table}", 0, table_end)
    table = tex[table_start:table_end]
    observed = re.findall(r"^(nnU-Net|SwinUNETR|UNETR|3D U-Net) & \$([BESD+]+)\$ & ([+\d.]+)", table, re.M)
    expected = [(row[0], row[1], f"{row[6]:+.2f}") for group in groups for row in group["rows"]]
    assert observed == expected, (observed, expected)

    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 8.5,
                         "mathtext.fontset": "stix", "pdf.fonttype": 42})
    blue = .85 * np.array([36, 105, 155]) / 255 + .15
    orange = .85 * np.array([205, 117, 42]) / 255 + .15
    gray, ink = "#6d7782", "#27313c"
    fig = plt.figure(figsize=(6.3, 5.8))
    left, width, height = .185, .53, .23
    combo_x, gain_x = .813, .948
    for group, name, bottom in zip(groups, ["(a) GeoFWI3D", "(b) MSD", "(c) ADNI"], [.725, .420, .115]):
        rows = group["rows"]
        ax = fig.add_axes([left, bottom, width, height])
        y = np.arange(4)
        for color, position, mean_idx, sd_idx in [(blue, y-.14, 2, 3), (orange, y+.14, 4, 5)]:
            ax.barh(position, [r[mean_idx] for r in rows], height=.22,
                    color=color, edgecolor="none", xerr=[r[sd_idx] for r in rows],
                    error_kw={"ecolor": ink, "elinewidth": .5,
                               "capsize": 1.6, "capthick": .5}, zorder=3)
        ax.set(xlim=(0, 100), ylim=(3.5, -.5), xticks=np.arange(0, 101, 20), yticks=y)
        ax.set_yticklabels([r[0] for r in rows], fontsize=9)
        ax.tick_params(axis="y", length=0, pad=8)
        ax.tick_params(axis="x", length=0, pad=4, colors=gray, labelsize=8)
        ax.grid(axis="x", color="#e1e3e5", linewidth=.5, zorder=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        title_y = bottom + height + .023
        fig.text(.01, title_y, name, fontsize=10, fontweight="bold", va="center")
        fig.text(combo_x, title_y, "Added terms", fontsize=8, color=gray, ha="center", va="center")
        fig.text(gain_x, title_y, "Gain (pp)", fontsize=8, color=gray, ha="center", va="center")
        for i, row in enumerate(rows):
            fy = bottom + height * (3.5 - i) / 4
            fig.text(combo_x, fy, "$" + row[1] + "$", fontsize=9,
                     ha="center", va="center")
            fig.text(gain_x, fy, f"{row[6]:+.2f}", fontsize=9,
                     ha="center", va="center")
    fig.text(left + width/2, .061, "Foreground Dice (%)", fontsize=9, ha="center", color=gray)
    fig.legend(handles=[Patch(facecolor=blue, label="Baseline"),
                        Patch(facecolor=orange, label="With spatial supervision")],
               loc="lower center", bbox_to_anchor=(.5, .005), ncol=2,
               frameon=False, fontsize=9, handlelength=1.1, columnspacing=3.)
    REPORT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, facecolor="white")
    plt.close(fig)
    record = dict(data_source=str(DATA.relative_to(ROOT)),
                  data_sha256=hashlib.sha256(DATA.read_bytes()).hexdigest(),
                  scores_and_dispersions_unchanged=True,
                  selected_combinations_checked_against="tab:paired-replication-summary",
                  rows=[dict(model=m, added_terms=c, gain_pp=float(g)) for m,c,g in expected],
                  output_sha256=hashlib.sha256(OUTPUT.read_bytes()).hexdigest())
    (REPORT / "provenance.json").write_text(json.dumps(record, indent=2) + "\n")
    print("Rendered twelve verified model/objective comparisons.")


if __name__ == "__main__":
    main()
