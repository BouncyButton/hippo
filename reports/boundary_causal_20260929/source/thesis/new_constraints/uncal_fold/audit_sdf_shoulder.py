"""Signed-distance curvature and upper/lower extrema audit for fold-0 A/P cuts."""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage
from sklearn.model_selection import KFold

from .audit_concavity_shoulder import (
    ROOT, SEED, ALPHAS, extract_shape, fit_parameters, predict, error_metrics, paired, sha,
    smooth_profile,
)
from .audit_predicted_foreground import largest_foreground_component

FAMILIES = ("sdf_ap", "sdf_3d", "extrema_depth", "sdf_extrema")


def signed_distance(mask: np.ndarray) -> np.ndarray:
    """Outside positive; caller supplies an explicit background pad."""
    return ndimage.distance_transform_edt(~mask) - ndimage.distance_transform_edt(mask)


def signed_curvatures(phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """AP level-curve curvature and divergence of outward 3-D unit normals."""
    gx, gy, gz = np.gradient(phi)
    gyy, gyz = np.gradient(gy, axis=1), np.gradient(gy, axis=2)
    gzz = np.gradient(gz, axis=2)
    denom = gy * gy + gz * gz
    ap = (gyy * gz * gz - 2 * gy * gz * gyz + gzz * gy * gy) / np.maximum(denom, 1e-6)**1.5
    ap[denom < 1e-4] = 0.0
    norm = np.sqrt(gx * gx + denom)
    nx, ny, nz = [g / np.maximum(norm, 1e-6) for g in (gx, gy, gz)]
    full = np.gradient(nx, axis=0) + np.gradient(ny, axis=1) + np.gradient(nz, axis=2)
    full[norm < 1e-4] = 0.0
    return ap, full


def envelope_depth(y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Perpendicular gap to the least concave majorant of a continuous profile."""
    if len(y) != len(z) or len(y) < 2 or np.any(np.diff(y) <= 0):
        raise ValueError("need at least two ordered profile points")
    hull: list[int] = []
    for i in range(len(y)):
        while len(hull) >= 2:
            a, b = hull[-2:]
            first = (z[b] - z[a]) / (y[b] - y[a])
            second = (z[i] - z[b]) / (y[i] - y[b])
            if first > second:
                break
            hull.pop()
        hull.append(i)
    depth = np.zeros(len(y))
    for a, b in zip(hull[:-1], hull[1:]):
        slope = (z[b]-z[a]) / (y[b]-y[a])
        roof = z[a] + slope * (y[a:b+1]-y[a])
        depth[a:b+1] = np.maximum(0.0, roof-z[a:b+1]) / np.sqrt(1+slope*slope)
    return depth


def extract_sdf_scores(mask: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
    """Binary-only geometry; return scores per y and exact extrema diagnostics."""
    if mask.ndim != 3 or mask.dtype != bool:
        raise ValueError("expected a Boolean [X,Y,Z] mask")
    mask, _, removed = largest_foreground_component(mask)
    pad = 6
    padded = np.pad(mask, pad)
    phi = ndimage.gaussian_filter(signed_distance(padded), 1.0)
    ap, full = signed_curvatures(phi)
    occupied = mask.any(axis=2)
    z = np.arange(mask.shape[2])
    upper = np.where(mask, z, -1).max(axis=2)
    lower = np.where(mask, z, mask.shape[2]).min(axis=2)
    xs, ys = np.where(occupied)
    coordinates = np.stack((xs+pad, ys+pad, upper[xs, ys]+pad+0.5))
    sample_ap = ndimage.map_coordinates(ap, coordinates, order=1, prefilter=False)
    sample_full = ndimage.map_coordinates(full, coordinates, order=1, prefilter=False)
    lower_coordinates = coordinates.copy()
    lower_coordinates[2] = lower[xs,ys]+pad-0.5
    sample_lower = ndimage.map_coordinates(ap, lower_coordinates, order=1, prefilter=False)
    count = np.bincount(ys, minlength=mask.shape[1]).astype(float)
    scores = {}
    for name, values in (("sdf_ap",sample_ap),("sdf_3d",sample_full)):
        concave = np.maximum(-values,0.0)
        mean = np.bincount(ys,weights=concave,minlength=len(count))/np.maximum(count,1)
        fraction = np.bincount(ys,weights=(values < -1e-4),minlength=len(count))/np.maximum(count,1)
        scores[name] = mean * fraction
    depth_sum, depth_count = np.zeros_like(count), np.zeros_like(count)
    for x in range(mask.shape[0]):
        runs,nruns = ndimage.label(occupied[x])
        for r in range(1,nruns+1):
            yy = np.flatnonzero(runs == r)
            if len(yy) < 8:
                continue
            u = ndimage.gaussian_filter1d(upper[x,yy].astype(float),1.0,mode="nearest")
            depth_sum[yy] += envelope_depth(yy.astype(float),u)
            depth_count[yy] += 1
    scores["extrema_depth"] = depth_sum/np.maximum(depth_count,1)
    # Smooth only across contiguous foreground; absent columns are never interpolated.
    for name, score in list(scores.items()):
        score[count == 0] = np.nan
        scores[name] = np.nan_to_num(smooth_profile(score),nan=0.0)
    scores["sdf_extrema"] = np.sqrt(scores["sdf_ap"] * scores["extrema_depth"])
    diagnostics = {"removed_voxels":removed,"occupied_columns":int(occupied.sum()),
                   "upper_inward_fraction":float(np.mean(sample_ap < -1e-4)),
                   "lower_inward_fraction":float(np.mean(sample_lower < -1e-4)),
                   "mean_column_thickness_mm":float(np.mean(upper[occupied]-lower[occupied]+1)),
                   "max_depth_mm":float(scores["extrema_depth"].max())}
    return scores,diagnostics


def shapes_for_mask(mask: np.ndarray, origin_y=0):
    base = extract_shape(mask,origin_y)
    profiles, diagnostics = extract_sdf_scores(mask)
    families={}
    c=base.candidates-origin_y
    for family, profile in profiles.items():
        shape=copy.copy(base)
        shape.consensus=0.5*(profile[c-1]+profile[c])
        families[family]=shape
    return families,diagnostics


def choose(names, family_shapes, targets, folds):
    errors={(f,a):[] for f in FAMILIES for a in ALPHAS}
    for fit_ix,test_ix in KFold(folds,shuffle=True,random_state=SEED).split(names):
        fitting=[names[i] for i in fit_ix]
        for family in FAMILIES:
            shapes=family_shapes[family]
            params=fit_parameters(fitting,shapes,targets)
            for i in test_ix:
                n=names[i]
                for alpha in ALPHAS:
                    errors[(family,alpha)].append(abs(predict(shapes[n],params,alpha)["hybrid"]-targets[n]))
    mae={key:float(np.mean(e)) for key,e in errors.items()}
    best=min(mae,key=lambda key:(mae[key],key[1],FAMILIES.index(key[0])))
    return best,{f"{f}/{a}":v for (f,a),v in mae.items()}


def main():
    out=ROOT/"evaluation_output/concavity_shoulder_fold0_20260923"
    base=json.loads((out/"summary.json").read_text())
    original=list(csv.DictReader((out/"case_results.csv").open()))
    train,val=base["frozen_model"]["train_ids"],base["frozen_model"]["val_ids"]
    dataset=ROOT/"datasets/Dataset101_MSD"
    targets={r["case"]:int(r["target"]) for r in original}
    family_shapes={f:{} for f in FAMILIES}
    diagnostics,labels={},{}
    for n in train+val:
        path=dataset/"labelsTr"/f"{n}.nii.gz"
        nii=nib.load(path)
        if not np.allclose(nii.affine[:3,:3],np.eye(3)):raise ValueError("1-mm RAS required")
        labels[n]=np.asarray(nii.dataobj)
        s,d=shapes_for_mask(labels[n]>0)
        for f in FAMILIES:family_shapes[f][n]=s[f]
        diagnostics[("reference",n)]=d
    print("SDF/extrema features extracted on 260 reference masks",flush=True)
    rows,fold_models=[],[]
    def add(n,subset,support,shapes,params,selection):
        family,alpha=selection
        r={"case":n,"split":subset,"support":support,"target":targets[n],
           "selected_family":family,"selected_alpha":alpha,**diagnostics[(support,n)]}
        for f in FAMILIES:
            p=predict(shapes[f],params[f],alpha if f==family else 0.0)
            r[f+"_raw"]=p["shoulder_raw"]
            r[f+"_offset"]=p["shoulder_offset"]
            r[f+"_zero_score"]=int(shapes[f].consensus.max() <= 1e-8)
            if f==family:
                r["selected"]=p["hybrid"];r["position"]=p["position"]
        rows.append(r)
    for fold,(fit_ix,test_ix) in enumerate(KFold(4,shuffle=True,random_state=SEED).split(train)):
        fitting=[train[i] for i in fit_ix]
        params={f:fit_parameters(fitting,family_shapes[f],targets) for f in FAMILIES}
        selection,cv=choose(fitting,family_shapes,targets,3)
        fold_models.append({"fold":fold,"fit_ids":fitting,"test_ids":[train[i] for i in test_ix],
                            "parameters":params,"selection":selection,"inner_mae":cv})
        for i in test_ix:
            n=train[i];add(n,"train_oof","reference",{f:family_shapes[f][n] for f in FAMILIES},params,selection)
    params={f:fit_parameters(train,family_shapes[f],targets) for f in FAMILIES}
    selection,cv=choose(train,family_shapes,targets,4)
    frozen={"parameters":params,"selection":selection,"training_cv_mae":cv,"outer_folds":fold_models}
    (out/"sdf_frozen_detector.json").write_text(json.dumps(frozen,indent=2)+"\n")
    print("Frozen SDF selection",selection,"offsets",{f:p["offset_mm"] for f,p in params.items()},flush=True)
    for n in val:add(n,"validation","reference",{f:family_shapes[f][n] for f in FAMILIES},params,selection)
    all_shapes={("reference",n):{f:family_shapes[f][n] for f in FAMILIES} for n in train+val}
    for support,inv in base["prediction_inventory"].items():
        for n in val:
            path=Path(inv["path"])/f"{n}.npz"
            if not path.exists():continue
            with np.load(path,allow_pickle=False) as cache:truth,pred=cache["ground_truth"],cache["prediction"]
            pads=[(int(a-b)//2,int(a-b)-int(a-b)//2) for a,b in zip(truth.shape,labels[n].shape)]
            if not np.array_equal(np.pad(labels[n],pads),truth):raise ValueError("label misalignment")
            s,d=shapes_for_mask(pred>0,-pads[1][0]);diagnostics[(support,n)]=d
            add(n,"validation",support,s,params,selection)
            old=next(r for r in original if r["case"]==n and r["support"]==support)
            rows[-1]["native_model"]=int(old["native_model"])
            all_shapes[(support,n)]=s
        print("Scored",support,flush=True)
    methods=["position",*[f+suffix for f in FAMILIES for suffix in ("_raw","_offset")],"selected"]
    groups={}
    for subset,support in sorted({(r["split"],r["support"]) for r in rows}):
        group=[r for r in rows if (r["split"],r["support"])==(subset,support)]
        target=[r["target"] for r in group]
        mm=methods+(["native_model"] if "native_model" in group[0] else [])
        groups[f"{subset}/{support}"]={
            "metrics":{m:error_metrics([r[m] for r in group],target) for m in mm},
            "paired":{f"{m}_vs_{comp}":paired([r[comp] for r in group],[r[m] for r in group],target)
                      for m in [*[f+"_offset" for f in FAMILIES],"selected"] for comp in ("position","native_model") if comp in group[0]},
            "zero_score_cases":{f:sum(r[f+"_zero_score"] for r in group) for f in FAMILIES}}
    summary={"source_sha256":sha(Path(__file__)),"schema":"sdf_extrema_shoulder.v1","exploratory":True,
             "frozen_model":frozen,"groups":groups,"source_audit":str(out/"summary.json")}
    (out/"sdf_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    with (out/"sdf_case_results.csv").open("w") as f:
        w=csv.DictWriter(f,fieldnames=sorted(set().union(*(r.keys() for r in rows))));w.writeheader();w.writerows(rows)
    with (out/"sdf_candidate_scores.csv").open("w") as f:
        w=csv.DictWriter(f,fieldnames=["case","support","candidate",*FAMILIES]);w.writeheader()
        for (support,n),shapes in all_shapes.items():
            for i,c in enumerate(shapes[FAMILIES[0]].candidates):
                w.writerow({"case":n,"support":support,"candidate":int(c),**{k:shapes[k].consensus[i] for k in FAMILIES}})
    report(out,summary,rows,all_shapes,labels)
    print(json.dumps({k:{m:round(v['mae_mm'],3) for m,v in g['metrics'].items()} for k,g in groups.items()},indent=2))


def report(out,summary,rows,all_shapes,labels):
    selection=summary["frozen_model"]["selection"]
    lines=["# Signed distance + extrema: fold-0 development audit","",
           "This follow-up uses the previous first/last occupied z representation. "
           "The SDF is negative inside and positive outside; concavity is negative signed curvature. "
           "Its value at the surface alone cannot localize a concavity. "
           "All geometry uses binary unions; no A/P classes enter feature extraction.","",
           "Four fixed families: AP curvature at upper extrema; 3-D curvature there; "
           "perpendicular gap to the upper convex-hull envelope; and the geometric mean "
           "of AP curvature and envelope depth. Signed-distance and profile smoothing are 1 mm. "
           "Lower extrema and thickness are diagnostic only. See the saved SDF protocol.","",
           f"Selected on full training CV: **{selection[0]}, position strength {selection[1]}**. "
           "Training rows use nested four-fold OOF estimates with three-fold inner family/strength selection.","",
           "| Set / foreground | Method | MAE mm | Within 1 mm | Within 2 mm | Max mm |","|---|---|---:|---:|---:|---:|"]
    for name,g in summary["groups"].items():
        for m,v in g["metrics"].items():
            lines.append(f"| {name} | {m} | {v['mae_mm']:.3f} | {v['within_1']:.1%} | {v['within_2']:.1%} | {v['max_mm']} |")
    lines += ["","## Paired comparisons","","| Set | Comparison | MAE change [95% crop bootstrap] | Helped / equal / harmed |","|---|---|---:|---:|"]
    for name,g in summary["groups"].items():
        for m,p in g["paired"].items():
            lo,hi=p['bootstrap95_mm']
            lines.append(f"| {name} | {m} | {p['second_minus_first_mae_mm']:+.3f} [{lo:+.3f}, {hi:+.3f}] | {p['helped']}/{p['equal']}/{p['harmed']} |")
    lines += ["","## Calibration and missing signals",""]
    for f,p in summary["frozen_model"]["parameters"].items():lines.append(f"- {f}: training offset {p['offset_mm']:+d} mm.")
    for name,g in summary["groups"].items():lines.append(f"- {name}: zero-score cases {g['zero_score_cases']}.")
    lines += ["","Zero-score maxima use a deterministic midpoint tie-break and are flagged; they are forced "
              "predictions, not detected landmarks. Hybrid selection includes a position prior, so controls are essential.","",
              "Reference foreground is oracle support. Fold 0 has been inspected repeatedly; this is development evidence. "
              "Participant pairing is unknown, so counts/bootstrap intervals are crop-level. All three prediction sets "
              "contain 52 validation and no training caches. No segmentation retrained or altered.","",
              "Reproduce: `MPLCONFIGDIR=/tmp/hippo-mpl .venv/bin/python -m thesis.new_constraints.uncal_fold.audit_sdf_shoulder`","",
              "[Every case](sdf_case_results.csv) · [Candidate scores](sdf_candidate_scores.csv) · [Frozen detector](sdf_frozen_detector.json)","",
              "![SDF comparison](sdf_overview.png)","","![Selected examples](sdf_examples.png)"]
    (out/"SDF_RESULTS.md").write_text("\n".join(lines)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .report_concavity_shoulder import draw_case,combined_overview
    fig,axes=plt.subplots(1,3,figsize=(16,5))
    for ax,key,title in zip(axes,("train_oof/reference","validation/reference","validation/early_augmented"),
                            ("Training OOF · reference","Validation · reference","Validation · augmented prediction")):
        g=summary['groups'][key]['metrics']
        names=["position",*[f+"_offset" for f in FAMILIES],"selected"]
        if "native_model" in g:names.append("native_model")
        values=[g[m]['mae_mm'] for m in names]
        ax.barh(names,values,color=['#9b80b5','#739cb7','#94afc0','#d39c58','#c37727','#008b80','#406176'][:len(names)])
        for i,v in enumerate(values):ax.text(v+.04,i,f"{v:.2f}",va='center',fontsize=9)
        ax.invert_yaxis();ax.set_xlim(0,max(values)*1.17);ax.set_xlabel('Mean absolute cut error (mm)');ax.set_title(title,fontsize=10)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Signed-distance curvature and extrema: does geometry locate the cut?',fontsize=14)
    fig.tight_layout();fig.savefig(out/'sdf_overview.png',dpi=150);plt.close(fig)
    selected=[]
    for subset in ('train_oof','validation'):
        g=sorted([r for r in rows if r['split']==subset and r['support']=='reference'],
                 key=lambda r:(abs(r['selected']-r['target'])-abs(r['position']-r['target']),r['case']))
        selected.extend(g[:3]+g[-3:])
    fig,axes=plt.subplots(4,3,figsize=(14,14))
    for ax,r in zip(axes.flat,selected):
        family=r['selected_family'];n=r['case'];s=all_shapes[('reference',n)][family]
        show={**r,'shoulder_raw':r[family+'_raw'],'hybrid':r['selected']}
        draw_case(ax,show,s,labels[n]>0,details=True)
        ax.set_title(f"{r['split']} · {n} · {family}\nprior {abs(r['position']-r['target'])} / selected {abs(r['selected']-r['target'])} mm",fontsize=8)
    fig.suptitle('Outcome-selected gains and harms versus position, both splits\nBlack: target · orange: raw geometric peak / score · green: selected · purple: position',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.95));fig.savefig(out/'sdf_examples.png',dpi=140);plt.close(fig)
    if (out/'broad_summary.json').exists():
        combined_overview(out,json.loads((out/'summary.json').read_text()),json.loads((out/'broad_summary.json').read_text()))
    index=(out/'index.html').read_text()
    if 'id="sdf-followup"' not in index:
        section='<section id="sdf-followup"><h2>Signed distance and extrema follow-up</h2><p>Training-only calibration, nested training OOF evaluation; development validation.</p><img src="sdf_overview.png"><p><a href="SDF_RESULTS.md">Full SDF results</a> · <a href="sdf_case_results.csv">Every case</a></p><details><summary>Selected examples</summary><img src="sdf_examples.png"></details></section>'
        index=index.replace('<img src="overview.png">',section+'<h2>Original local-shoulder audit</h2><img src="overview.png">')
        (out/'index.html').write_text(index)


if __name__ == "__main__":main()
