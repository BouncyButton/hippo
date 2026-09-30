#!/usr/bin/env python3
"""Can the prediction's OWN sagittal step fit locate and fix its errors?

For each sagittal section x and each occupied y column, take the predicted
upper (max z) and lower (min z) foreground extent. Fit decreasing step
functions over y within contiguous predicted runs (no labels used), round,
and snap each column's extent to the fit. Every voxel the snap adds/removes is
scored against the reference: precision = fraction of changes that fix an error.
Numpy only (own PAV).
"""
import sys, glob, json
import numpy as np

def pav_dec(v):
    blocks = []  # [sum, count]
    for x in v:
        blocks.append([x, 1])
        while len(blocks) > 1 and blocks[-2][0]/blocks[-2][1] < blocks[-1][0]/blocks[-1][1]:
            s, c = blocks.pop(); blocks[-1][0] += s; blocks[-1][1] += c
    out = []
    for s, c in blocks: out += [s/c]*c
    return np.array(out)

def runs(mask1d):
    idx = np.flatnonzero(mask1d); out=[];
    if len(idx)==0: return out
    start=idx[0]; prev=idx[0]
    for i in idx[1:]:
        if i!=prev+1: out.append((start,prev)); start=i
        prev=i
    out.append((start,prev)); return out

def analyse(pred, ref, min_run=3, tol=1):
    corrected = pred.copy()
    add_ok=add_bad=rem_ok=rem_bad=0
    cols_changed=0; cols=0
    X,Y,Z = pred.shape
    for x in range(X):
        occ = pred[x].any(axis=1)
        for a,b in runs(occ):
            if b-a+1 < min_run: continue
            ys = np.arange(a,b+1)
            up = np.array([np.flatnonzero(pred[x,y]).max() for y in ys], float)
            lo = np.array([np.flatnonzero(pred[x,y]).min() for y in ys], float)
            fu = np.rint(pav_dec(up)).astype(int); fl = np.rint(pav_dec(lo)).astype(int)
            for k,y in enumerate(ys):
                cols += 1; changed=False
                for raw, fit, side in ((int(up[k]),fu[k],'up'),(int(lo[k]),fl[k],'lo')):
                    if abs(fit-raw) < tol: continue
                    changed=True
                    if side=='up':
                        zs = range(raw+1, fit+1) if fit>raw else range(fit+1, raw+1)
                        add = fit>raw
                    else:
                        zs = range(fit, raw) if fit<raw else range(raw, fit)
                        add = fit<raw
                    for z in zs:
                        if not (0<=z<Z): continue
                        if add:
                            if ref[x,y,z]: add_ok+=1
                            else: add_bad+=1
                            corrected[x,y,z]=True
                        else:
                            if not ref[x,y,z]: rem_ok+=1
                            else: rem_bad+=1
                            corrected[x,y,z]=False
                cols_changed += changed
    return corrected, dict(add_ok=add_ok,add_bad=add_bad,rem_ok=rem_ok,rem_bad=rem_bad,cols=cols,cols_changed=cols_changed)

def extreme_error_share(pred, ref):
    """Fraction of error voxels that sit at a column's z-extreme (top/bottom) in pred or ref."""
    err = pred ^ ref
    ext = np.zeros_like(err)
    for m in (pred, ref):
        occ = m.any(axis=2)
        zs = np.arange(m.shape[2])
        top = np.where(m, zs, -1).max(axis=2); bot = np.where(m, zs, 999).min(axis=2)
        for z in range(m.shape[2]):
            ext[:,:,z] |= occ & ((top==z)|(bot==z))
    # error voxels outside [bot,top] of ref or pred extent also count as extreme-region
    return int(err.sum()), int((err & ext).sum())

def dice(a,b): return 2*(a&b).sum()/(a.sum()+b.sum())

if __name__=='__main__':
    root = sys.argv[1]; tol = int(sys.argv[2]) if len(sys.argv)>2 else 1
    tot = dict(add_ok=0,add_bad=0,rem_ok=0,rem_bad=0,cols=0,cols_changed=0)
    d0=[];d1=[]; errs=0; ext=0; improved=worse=0
    for f in sorted(glob.glob(root+'/*.npz')):
        d=np.load(f); pred=d['prediction']>0; ref=d['ground_truth']>0
        corr, s = analyse(pred, ref, tol=tol)
        for k in tot: tot[k]+=s[k]
        a,b = dice(pred,ref), dice(corr,ref); d0.append(a); d1.append(b)
        improved += b>a+1e-9; worse += b<a-1e-9
        e, x = extreme_error_share(pred, ref); errs+=e; ext+=x
    changes = tot['add_ok']+tot['add_bad']+tot['rem_ok']+tot['rem_bad']
    fixed = tot['add_ok']+tot['rem_ok']
    print(json.dumps(dict(
        tol=tol, cases=len(d0), total_error_voxels=errs,
        error_voxels_at_column_z_extremes=ext, extreme_share=round(ext/errs,3),
        columns=tot['cols'], columns_changed=tot['cols_changed'],
        voxels_changed=changes, fixed=fixed, broken=changes-fixed,
        precision=round(fixed/max(changes,1),3),
        share_of_all_errors_fixed=round(fixed/errs,4),
        add=[tot['add_ok'],tot['add_bad']], remove=[tot['rem_ok'],tot['rem_bad']],
        dice_before=round(float(np.mean(d0)),4), dice_after=round(float(np.mean(d1)),4),
        cases_improved=int(improved), cases_worse=int(worse)), indent=1))
