"""Static figures and complete galleries for the concavity-shoulder audit."""
from __future__ import annotations

import html
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"target": "#20252b", "shoulder_raw": "#cb7119", "hybrid": "#008b80", "position": "#8b58aa"}


def combined_overview(out, local, broad):
    """Compare both audit families without hiding the original negative result."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, key, title in zip(axes, ("train_oof/reference", "validation/reference", "validation/early_augmented"),
                              ("Training OOF · reference mask", "Validation · reference mask", "Validation · augmented prediction")):
        first, second = local["groups"][key]["metrics"], broad["groups"][key]["metrics"]
        names = ["Position only", "Local shoulder + offset", "Broad shoulder + offset*", "Local + position", "Broad + position*"]
        vals = [first["position"]["mae_mm"], first["shoulder_offset"]["mae_mm"], second["broad_offset"]["mae_mm"], first["hybrid"]["mae_mm"], second["broad_hybrid"]["mae_mm"]]
        if "native_model" in first:
            names.append("Model's own cut"); vals.append(first["native_model"]["mae_mm"])
        ax.barh(names, vals, color=["#9b80b5", "#cc945b", "#c16b1d", "#65ada3", "#008b80", "#557c9e"][:len(vals)])
        for i, value in enumerate(vals): ax.text(value+.04, i, f"{value:.2f}", va="center", fontsize=9)
        ax.invert_yaxis(); ax.set_xlim(0,max(vals)*1.17); ax.set_xlabel("Mean absolute cut error (mm)")
        ax.set_title(title, fontsize=10); ax.spines[["top","right"]].set_visible(False)
    fig.suptitle("Shoulder geometry does not improve cut localization in this audit",fontsize=15)
    fig.text(.5,.025,"* Broad detector: exploratory follow-up. All offsets and position weights fitted on training only.",ha="center",fontsize=10)
    fig.tight_layout(rect=(0,.05,1,.95));fig.savefig(out/"combined_overview.png",dpi=150);plt.close(fig)


def draw_case(ax, row, shape, label, *, details=False):
    ax.imshow(label.any(axis=0).T, origin="lower", cmap="Greys", alpha=0.3, interpolation="nearest")
    ax.plot(shape.profile_y, shape.profile_z, color="#2676ae", lw=1.5)
    for key in ("target", "shoulder_raw", "hybrid", "position"):
        ax.axvline(row[key] - 0.5, color=COLORS[key], lw=1.4, ls="--" if key in ("target", "position") else "-")
    ax.set_xlim(shape.high + 3, shape.low - 3)
    z = shape.profile_z[np.isfinite(shape.profile_z)]
    ax.set_ylim(max(0, np.min(np.argwhere(label)[:, 2]) - 2), z.max() + 4)
    ax.set_title(f"{row['case']} | raw {abs(row['shoulder_raw']-row['target'])} mm; hybrid {abs(row['hybrid']-row['target'])} mm", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.set_xlabel("native y (anterior ←)", fontsize=8)
    if details:
        # twinx requires adjustable data limits for equal aspect, which can crop
        # the top contour. Fix the physical panel ratio instead of its data aspect.
        ax.set_aspect("auto")
        ax.set_box_aspect((ax.get_ylim()[1] - ax.get_ylim()[0]) / (shape.high - shape.low + 6))
        score_ax = ax.twinx()
        score_ax.set_box_aspect(ax.get_box_aspect())
        score_ax.plot(shape.candidates - 0.5, shape.consensus, color=COLORS["shoulder_raw"], alpha=0.5, lw=1)
        score_ax.set_ylim(0, max(1.0, shape.consensus.max() * 1.25))
        score_ax.set_ylabel("shoulder score", fontsize=8)
        score_ax.tick_params(labelsize=7)


def write_report(out: Path, summary: dict, rows: list[dict], shapes: dict, labels: dict, dataset_root: Path):
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white"})
    groups = summary["groups"]
    title = "Concavity-shoulder localization — fold 0"
    lines = [f"# {title}", "", "208 training crops (nested out-of-fold estimates), 52 validation crops. "
             "No segmentation network retrained. All distances are millimetres.", "",
             "## Localization results", "",
             "| Set / foreground | Method | MAE | Within 1 mm | Within 2 mm | p90 | Maximum |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for group_name, group in groups.items():
        for method, m in group["metrics"].items():
            lines.append(f"| {group_name} | {method} | {m['mae_mm']:.3f} | {m['within_1']:.1%} | {m['within_2']:.1%} | {m['p90_mm']:.1f} | {m['max_mm']} |")
    lines += ["", "## Paired comparisons", "", "Negative changes favour the shoulder method. "
              "Bootstrap intervals resample crops (5,000 replicates); exploratory, not participant-level inference.", "",
              "| Set / foreground | Comparator | Method | MAE change [95% interval] | Helped / equal / harmed |",
              "|---|---|---|---:|---:|"]
    for name, group in groups.items():
        for comp in ("position", "model"):
            for method, p in group[f"paired_vs_{comp}"].items():
                lo, hi = p["bootstrap95_mm"]
                lines.append(f"| {name} | {comp} | {method} | {p['second_minus_first_mae_mm']:+.3f} [{lo:+.3f}, {hi:+.3f}] | {p['helped']}/{p['equal']}/{p['harmed']} |")
    model = summary["frozen_model"]
    lines += ["", "## Frozen training fit", "",
              f"Median shoulder-to-cut correction: **{model['parameters']['offset_mm']:+d} mm**. "
              f"Hybrid positional penalty strength: **{model['alpha']}** (selected on training CV).",
              f"Training relative cut median: {model['parameters']['relative_median']:.4f}; "
              f"robust scale: {model['parameters']['relative_scale']:.4f}.", "",
              "The hybrid can exploit position heavily. Its result alone does not demonstrate that a "
              "recognizable concavity independently identifies the boundary. Raw, offset-corrected and position-only controls are essential.", "",
              "## Descriptive uncertainty intervals", "",
              "Radii use the 90th percentile (higher order statistic) of training-OOF absolute errors. "
              "These are not calibrated per-case confidence intervals or formal conformal guarantees.", "",
              "| Method | Radius | Training OOF coverage | Validation reference coverage |",
              "|---|---:|---:|---:|"]
    for m, radius in summary["interval_radius_mm"].items():
        lines.append(f"| {m} | ±{radius:.0f} mm | {groups['train_oof/reference']['interval_coverage'][m]:.1%} | {groups['validation/reference']['interval_coverage'][m]:.1%} |")
    lines += ["", "## Geometry and provenance", ""]
    for name, group in groups.items():
        lines.append(f"- {name}: {group['nonplanar_cases']} nonplanar references; "
                     f"{group['reference_ties']} tied reference cuts; {group['zero_shoulder_cases']} zero-score shoulders; "
                     f"{group['target_outside_detector_range']} targets outside the detector's complete-window range.")
    lines += ["", "The detector uses sigma=1 mm smoothing, two four-sample local slopes, and the "
              "geometric mean of silhouette and sagittal-slice consensus scores. It sees binary "
              "foreground only. A/P classes enter calibration and scoring, never shape extraction.", "",
              "For predictions, cached reference masks were checked against exact center-padded native labels; "
              "full prediction volumes were used, without cropping off predictions outside native image bounds. "
              "Only the largest 26-connected component enters shape extraction; original masks are not modified.", "",
              "Saved prediction availability:"]
    for support, inv in summary["prediction_inventory"].items():
        lines.append(f"- {support}: {inv['train_available']}/208 training and {inv['val_available']}/52 validation caches.")
    lines += ["", *[f"- {w}" for w in summary["warnings"]], "",
              "## Reproduce", "", "```bash",
              "MPLCONFIGDIR=/tmp/hippo-mpl .venv/bin/python -m thesis.new_constraints.uncal_fold.audit_concavity_shoulder",
              ".venv/bin/python -m pytest thesis/new_constraints/uncal_fold/test_concavity_shoulder.py -q", "```", "",
              "## Artifacts", "", "[Complete visual gallery](index.html) · [Per-case results](case_results.csv) · "
              "[Candidate scores](candidate_scores.csv) · [Summary](summary.json) · [Frozen detector](frozen_detector.json) · "
              "[Input hashes](input_manifest.json)", "", "![Overview](overview.png)"]
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, key, title_part in zip(axes, ["train_oof/reference", "validation/reference", "validation/early_augmented"],
                                  ["Training (OOF), reference mask", "Validation, reference mask", "Validation, predicted mask"]):
        group = groups[key]["metrics"]
        methods = ["position", "shoulder_raw", "shoulder_offset", "hybrid"]
        if "native_model" in group:
            methods.append("native_model")
        vals = [group[m]["mae_mm"] for m in methods]
        ax.barh(methods, vals, color=["#9b80b5", "#cf9b65", "#c97121", "#008b80", "#547d9e"][:len(methods)])
        for i, v in enumerate(vals): ax.text(v + 0.04, i, f"{v:.2f}", va="center", fontsize=9)
        ax.invert_yaxis(); ax.set_xlabel("Mean absolute cut error (mm)")
        ax.set_xlim(0, max(vals) * 1.22); ax.set_title(title_part, fontsize=11)
    fig.suptitle("Can the concavity shoulder locate the A/P cut?", fontsize=15)
    fig.tight_layout(); fig.savefig(out / "overview.png", dpi=160); plt.close(fig)

    gallery = []
    for subset in ("train_oof", "validation"):
        group = sorted([r for r in rows if r["split"] == subset and r["support"] == "reference"], key=lambda r: r["case"])
        for start in range(0, len(group), 12):
            page = group[start:start + 12]
            fig, axes = plt.subplots(4, 3, figsize=(13, 13))
            for ax, row in zip(axes.flat, page):
                draw_case(ax, row, shapes[("reference", row["case"])], labels[row["case"]] > 0)
            for ax in list(axes.flat)[len(page):]: ax.axis("off")
            fig.suptitle(f"{subset}: all reference unions, page {start//12+1}\nBlack dashed: reference cut · orange: raw shoulder · green: hybrid · purple dashed: position", fontsize=12)
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            filename = f"{subset}_{start//12+1:02d}.png"
            fig.savefig(out / filename, dpi=120); plt.close(fig)
            gallery.append((subset, filename))
    # Explicit outcome-selected examples, not a representative accuracy sample.
    selected = []
    for subset in ("train_oof", "validation"):
        group = sorted([r for r in rows if r["split"] == subset and r["support"] == "reference"],
                       key=lambda r: (abs(r["shoulder_raw"] - r["target"]), r["case"]))
        selected.extend(group[:3] + group[-3:])
    fig, axes = plt.subplots(4, 3, figsize=(14, 14))
    for ax, row in zip(axes.flat, selected):
        draw_case(ax, row, shapes[("reference", row["case"])], labels[row["case"]] > 0, details=True)
        ax.set_title(row["split"] + " · " + ax.get_title(), fontsize=8)
    fig.suptitle("Outcome-selected examples: three smallest and three largest raw-shoulder errors per split\nBlack: reference · orange: shoulder / score · green: hybrid · purple: position", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(out / "examples.png", dpi=140); plt.close(fig)

    # Frozen detector on real predicted geometry, with native-model cut overlaid.
    for support in summary["prediction_inventory"]:
        group = sorted([r for r in rows if r["support"] == support],
                       key=lambda r: (abs(r["hybrid"]-r["target"])-abs(r["native_model"]-r["target"]), r["case"]))
        if not group: continue
        chosen = group[:3] + group[-3:]
        fig, axes = plt.subplots(2, 3, figsize=(13, 8))
        for ax, row in zip(axes.flat, chosen):
            n = row["case"]
            s = shapes[(support, n)]
            # Gray native reference silhouette is context; blue line is predicted contour.
            draw_case(ax, row, s, labels[n] > 0, details=True)
            ax.axvline(row["native_model"]-.5, color="#547d9e", ls=":", lw=2)
            ax.set_title(f"{n} · model {abs(row['native_model']-row['target'])} mm / hybrid {abs(row['hybrid']-row['target'])} mm", fontsize=9)
        fig.suptitle(f"{support}: outcome-selected gains (top) and harms (bottom)\nGray: reference silhouette · blue: predicted contour / dotted model cut · orange: shoulder · green: hybrid", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        filename = f"{support}_examples.png"
        fig.savefig(out / filename, dpi=140); plt.close(fig)
        gallery.append((support, filename))
    links = "".join(f'<p><a href="{filename}">{html.escape(subset)} — {filename}</a></p>' for subset, filename in gallery)
    (out / "index.html").write_text(f'''<!doctype html><meta charset="utf-8"><title>{title}</title>
<style>body{{font:16px system-ui;max-width:1150px;margin:40px auto;padding:0 20px;color:#253342}}img{{max-width:100%}}a{{color:#076c88}}summary{{cursor:pointer;padding:12px}}</style>
<h1>{title}</h1><p>208 training crops, 52 validation crops. Training metrics use nested out-of-fold predictions.
Reference masks are oracle support. Validation is previously examined development data. No network retrained.</p>
<p><a href="RESULTS.md">Full results</a> · <a href="case_results.csv">Every case (CSV)</a> · <a href="candidate_scores.csv">Every candidate score</a> · <a href="summary.json">Summary JSON</a></p>
<img src="overview.png"><h2>Selected successes and failures</h2><p>Outcome-selected, not an accuracy sample.</p><img src="examples.png">
<h2>All-case reference galleries and prediction examples</h2>{links}
''' + "".join(f'<details><summary>{html.escape(subset)}: {filename}</summary><img loading="lazy" src="{filename}"></details>' for subset, filename in gallery))
