"""Exploratory broad-scale follow-up to the fixed local-shoulder audit.

Motivated by training contour inspection after seeing initial validation metrics.
Both detector families must be reported; this is not a fresh confirmatory test.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from sklearn.model_selection import KFold

from .audit_concavity_shoulder import (
    ROOT, SEED, extract_shape, fit_parameters, predict, select_alpha,
    error_metrics, paired, sha,
)


def broad_score(y: np.ndarray, z: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Continuous two-line fit; improvement over one line locates a broad elbow.

    Remove 3 mm at each end to avoid cap curvature. Candidate hinge is c-0.5;
    require four samples on each side, a descending posterior segment, and a
    positive slope change. No target labels or position prior enter this score.
    """
    valid = np.isfinite(z)
    y, z = y[valid].astype(float), z[valid]
    if len(y) < 14:
        return np.zeros(len(candidates))
    interior = (y >= y.min()+3) & (y <= y.max()-3)
    y, z = y[interior], z[interior]
    linear = np.column_stack((np.ones(len(y)), y-y.mean()))
    residual = z - linear @ np.linalg.lstsq(linear, z, rcond=None)[0]
    baseline_sse = float(residual @ residual)
    scores = np.zeros(len(candidates))
    if baseline_sse < 1e-8:
        return scores
    for i, c in enumerate(candidates):
        x = y - (c-0.5)
        if np.sum(x < 0) < 4 or np.sum(x >= 0) < 4:
            continue
        design = np.column_stack((np.ones(len(x)), x, np.maximum(x, 0)))
        beta = np.linalg.lstsq(design, z, rcond=None)[0]
        if beta[1] >= 0 or beta[2] <= 0:
            continue
        residual = z - design @ beta
        scores[i] = max(0.0, 1 - float(residual @ residual) / baseline_sse)
    return scores


def broad_shape(mask, origin_y=0):
    shape = extract_shape(mask, origin_y)
    shape.consensus = broad_score(shape.profile_y, shape.profile_z, shape.candidates)
    return shape


def main():
    out = ROOT / "evaluation_output/concavity_shoulder_fold0_20260923"
    base = json.loads((out / "summary.json").read_text())
    old_rows = list(csv.DictReader((out / "case_results.csv").open()))
    train, val = base["frozen_model"]["train_ids"], base["frozen_model"]["val_ids"]
    dataset = ROOT / "datasets/Dataset101_MSD"
    targets = {r["case"]: int(r["target"]) for r in old_rows}
    labels = {n: np.asarray(nib.load(dataset / "labelsTr" / f"{n}.nii.gz").dataobj) for n in train+val}
    shapes = {n: broad_shape(label > 0) for n, label in labels.items()}
    rows, folds = [], []
    def row(n, split, support, s, params, alpha):
        predictions = predict(s, params, alpha)
        return {"case": n, "split": split, "support": support, "target": targets[n],
                "position": predictions["position"], "broad_raw": predictions["shoulder_raw"],
                "broad_offset": predictions["shoulder_offset"], "broad_hybrid": predictions["hybrid"],
                "fit_improvement": float(s.consensus.max())}
    for i, (fit_ix, test_ix) in enumerate(KFold(4, shuffle=True, random_state=SEED).split(train)):
        fitting = [train[j] for j in fit_ix]
        params = fit_parameters(fitting, shapes, targets)
        alpha, cv = select_alpha(fitting, shapes, targets, 3)
        folds.append({"fold": i, "parameters": params, "alpha": alpha, "cv": cv})
        rows.extend(row(train[j], "train_oof", "reference", shapes[train[j]], params, alpha) for j in test_ix)
    params = fit_parameters(train, shapes, targets)
    alpha, cv = select_alpha(train, shapes, targets, 4)
    frozen = {"parameters": params, "alpha": alpha, "training_cv": cv, "oof_fits": folds}
    (out / "broad_frozen_detector.json").write_text(json.dumps(frozen, indent=2)+"\n")
    rows.extend(row(n, "validation", "reference", shapes[n], params, alpha) for n in val)
    all_shapes = {("reference", n): s for n, s in shapes.items()}
    for support, inv in base["prediction_inventory"].items():
        for n in val:
            path = Path(inv["path"]) / f"{n}.npz"
            if not path.exists(): continue
            with np.load(path, allow_pickle=False) as cache:
                truth, pred = cache["ground_truth"], cache["prediction"]
            pads = [(int(a-b)//2, int(a-b)-int(a-b)//2) for a,b in zip(truth.shape, labels[n].shape)]
            assert np.array_equal(np.pad(labels[n], pads), truth)
            s = broad_shape(pred > 0, -pads[1][0])
            r = row(n, "validation", support, s, params, alpha)
            original = next(r for r in old_rows if r["case"] == n and r["support"] == support)
            r["native_model"] = int(original["native_model"])
            rows.append(r); all_shapes[(support,n)] = s
    groups = {}
    for split, support in sorted({(r["split"],r["support"]) for r in rows}):
        group = [r for r in rows if (r["split"],r["support"]) == (split,support)]
        target = [r["target"] for r in group]
        methods = ["position", "broad_raw", "broad_offset", "broad_hybrid"] + (["native_model"] if "native_model" in group[0] else [])
        groups[f"{split}/{support}"] = {"metrics": {m:error_metrics([r[m] for r in group],target) for m in methods},
              "paired": {f"{m}_vs_{b}":paired([r[b] for r in group],[r[m] for r in group],target)
                         for m in ("broad_offset","broad_hybrid") for b in ("position","native_model") if b in group[0]},
              "zero_bend_cases":sum(r["fit_improvement"] <= 1e-8 for r in group)}
    summary = {"source_sha256":sha(Path(__file__)), "exploratory_followup":True, "frozen_model":frozen,"groups":groups}
    (out / "broad_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    with (out / "broad_case_results.csv").open("w") as f:
        w=csv.DictWriter(f,fieldnames=sorted(set().union(*(r.keys() for r in rows))))
        w.writeheader();w.writerows(rows)
    lines=["# Broad-shoulder exploratory follow-up", "",
           "This second definition was added after inspecting the original local-score results and training "
           "outlines. No validation-fitted parameter is used, but the analysis is explicitly exploratory. "
           "The original detector and its negative results remain unchanged.", "",
           "Fit a continuous pair of straight lines to the smoothed upper sagittal silhouette, "
           "excluding 3 mm at both tips. Rank the hinge by reduction in residual sum of squares "
           "relative to one line. Require a descending posterior slope and a positive slope change. "
           "The hinge is the broad shoulder. This removes the first detector's sensitivity to small local bends.", "",
           f"Training-median cut offset: {params['offset_mm']:+d} mm. Position strength: {alpha}. "
           "Training uses four outer folds with three-fold inner selection; final selection uses training-only four-fold CV.", "",
           "| Set / support | Method | MAE mm | Within 1 mm | Within 2 mm | Max mm |", "|---|---|---:|---:|---:|---:|"]
    for k,g in groups.items():
        for m,v in g["metrics"].items():
            lines.append(f"| {k} | {m} | {v['mae_mm']:.3f} | {v['within_1']:.1%} | {v['within_2']:.1%} | {v['max_mm']} |")
    lines += ["", "| Set / support | Comparison | Change in MAE [95% crop bootstrap] | Helped / equal / harmed |", "|---|---|---:|---:|"]
    for k,g in groups.items():
        for m,p in g["paired"].items():
            lo,hi=p["bootstrap95_mm"]
            lines.append(f"| {k} | {m} | {p['second_minus_first_mae_mm']:+.3f} [{lo:+.3f}, {hi:+.3f}] | {p['helped']}/{p['equal']}/{p['harmed']} |")
    lines += ["", "Reproduce: `.venv/bin/python -m thesis.new_constraints.uncal_fold.audit_broad_shoulder`", "",
              "![Outcome-selected broad-shoulder examples](broad_examples.png)"]
    (out/"BROAD_RESULTS.md").write_text("\n".join(lines)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .report_concavity_shoulder import draw_case, combined_overview
    selected=[]
    for split in ("train_oof","validation"):
        group=sorted([r for r in rows if r["split"]==split and r["support"]=="reference"],key=lambda r:(abs(r['broad_raw']-r['target']),r['case']))
        selected.extend(group[:3]+group[-3:])
    fig,axes=plt.subplots(4,3,figsize=(14,14))
    for ax,r in zip(axes.flat,selected):
        n=r["case"]
        show={**r,"shoulder_raw":r["broad_raw"],"hybrid":r["broad_hybrid"]}
        draw_case(ax,show,all_shapes[("reference",n)],labels[n]>0,details=True)
        ax.set_title(r["split"]+" · "+ax.get_title(),fontsize=8)
    fig.suptitle("Exploratory broad shoulder: smallest / largest raw errors per split\nOrange: broad hinge / fit score · black: reference · green: hybrid · purple: position",fontsize=12)
    fig.tight_layout(rect=(0,0,1,.95));fig.savefig(out/"broad_examples.png",dpi=140);plt.close(fig)
    combined_overview(out, base, summary)
    index = (out/"index.html").read_text()
    if 'id="broad-followup"' not in index:
        index = index.replace('<img src="overview.png">', '<section id="broad-followup"><h2>Combined result</h2><p>The broad fit is an explicitly exploratory follow-up. Neither detector improves on the position prior or the models\' own cuts.</p><img src="combined_overview.png"><p><a href="BROAD_RESULTS.md">Broad-fit results</a> · <a href="broad_case_results.csv">Broad-fit per-case CSV</a></p><details><summary>Broad-fit selected examples</summary><img src="broad_examples.png"></details></section><h2>Original local detector</h2><img src="overview.png">')
        (out/"index.html").write_text(index)
    print(json.dumps({k:{m:round(v['mae_mm'],3) for m,v in g['metrics'].items()} for k,g in groups.items()},indent=2))


if __name__ == "__main__": main()
