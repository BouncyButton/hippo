#!/usr/bin/env python3
"""Step models = rasterised smooth surfaces. For the upper/lower z-extreme height
maps h(x,y), fit a smooth surface and round it to voxels. Report
 (a) capacity: fraction of REFERENCE column extremes reproduced when fitting the reference itself
 (b) correction: fitting the PREDICTION, how many wrong extremes it fixes vs correct ones it breaks."""
import sys, glob, numpy as np
sys.path.insert(0,'scripts')
from audit_self_step_correction import pav_dec, runs

def nan_gauss(H, sx, sy):
    from scipy.ndimage import gaussian_filter
    m=~np.isnan(H); V=np.where(m,H,0.0)
    num=gaussian_filter(V,(sx,sy),mode='constant'); den=gaussian_filter(m.astype(float),(sx,sy),mode='constant')
    return np.where(m, num/np.maximum(den,1e-6), np.nan)

def per_section(H, fn):
    out=np.full_like(H,np.nan)
    for x in range(H.shape[0]):
        occ=~np.isnan(H[x])
        for a,b in runs(occ):
            ys=np.arange(a,b+1)
            out[x,ys]=fn(ys,H[x,ys]) if b-a+1>=3 else H[x,ys]
    return out

def poly(deg):
    def f(ys,v):
        d=min(deg,len(ys)-1); return np.polyval(np.polyfit(ys,v,d),ys)
    return f

MODELS={
 'isotonic (1D)':lambda H: per_section(H, lambda y,v: pav_dec(v)),
 'line per section':lambda H: per_section(H, poly(1)),
 'quadratic per section':lambda H: per_section(H, poly(2)),
 'cubic per section':lambda H: per_section(H, poly(3)),
 'gauss 2D sx=1,sy=1':lambda H: nan_gauss(H,1,1),
 'gauss 2D sx=1,sy=2':lambda H: nan_gauss(H,1,2),
 'gauss 2D sx=2,sy=2':lambda H: nan_gauss(H,2,2),
 'gauss 1D along y sy=1.5':lambda H: nan_gauss(H,0.01,1.5),
}
def heights(m, side):
    zs=np.arange(m.shape[2]); o=m.any(2)
    v=np.where(m,zs,-1).max(2) if side=='top' else np.where(m,zs,999).min(2)
    return np.where(o,v,np.nan).astype(float)

root=sys.argv[1]
files=sorted(glob.glob(root+'/*.npz'))
stats={k:dict(cap=0,capn=0,fix=0,brk=0,wrong=0) for k in MODELS}
for f in files:
    d=np.load(f); p=d['prediction']>0; r=d['ground_truth']>0
    for side in ('top','bot'):
        R=heights(r,side); P=heights(p,side); both=~np.isnan(R)&~np.isnan(P)
        for k,M in MODELS.items():
            s=stats[k]
            FR=np.rint(M(R)); okR=~np.isnan(R)
            s['cap']+=int((FR[okR]==R[okR]).sum()); s['capn']+=int(okR.sum())
            FP=np.rint(M(P))
            raw_ok=(P==R)&both; new_ok=(FP==R)&both
            s['fix']+=int((~raw_ok&new_ok).sum()); s['brk']+=int((raw_ok&~new_ok).sum()); s['wrong']+=int((~raw_ok&both).sum())
wr=stats['isotonic (1D)']['wrong']
print(f'wrong predicted column extremes: {wr}')
print(f"{'model':28s} {'ref reproduced':>14s} {'fixes':>7s} {'breaks':>7s} {'net':>7s}")
for k,s in stats.items():
    print(f"{k:28s} {s['cap']/s['capn']:14.3f} {s['fix']:7d} {s['brk']:7d} {s['fix']-s['brk']:7d}")
