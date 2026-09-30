#!/usr/bin/env python3
"""At column z-extremes where prediction and reference disagree, does the MRI
intensity favour the reference edge? Two tests:
 A) edge strength: |I(e+1)-I(e)| at the reference interface vs the predicted one
 B) region likeness: is each error voxel's intensity closer to the case's
    reference-inside mean or its outside-shell mean (Mumford-Shah style)?"""
import sys, glob, numpy as np, nibabel as nib
from scipy import ndimage
root=sys.argv[1]; img_dir='datasets/Dataset101_MSD/imagesTr'
A_ref=A_pred=A_tie=0; B_ok=B_bad=0; rank_hits=np.zeros(5); rank_n=0
for f in sorted(glob.glob(root+'/*.npz')):
    name=f.split('/')[-1][:-4]
    d=np.load(f); gt=d['ground_truth']; p=d['prediction']>0; r=gt>0
    im=nib.load(f'{img_dir}/{name}_0000.nii.gz').get_fdata().astype(np.float32)
    delta=np.array(gt.shape)-np.array(im.shape); b=delta//2
    im=np.pad(im,tuple(zip(b,delta-b)),mode='edge')
    im=(im-im[r].mean())/(im[r].std()+1e-6)
    zs=np.arange(64)
    for side in ('top','bot'):
        def ext(m):
            return (np.where(m,zs,-1).max(2) if side=='top' else np.where(m,zs,999).min(2))
        R=ext(r);P=ext(p);occ=r.any(2)&p.any(2)
        for x,y in zip(*np.nonzero(occ)):
            prof=im[x,y]
            def grad(e):  # interface just outside the extreme
                o=e+1 if side=='top' else e-1
                return abs(prof[o]-prof[e]) if 0<=o<64 else np.nan
            g=R[x,y]
            cand=[grad(g+k) for k in (-2,-1,0,1,2)]
            if not np.isnan(cand).any():
                rank_hits[int(np.argmax(cand))]+=1; rank_n+=1
            if P[x,y]!=g:
                a,c=grad(g),grad(P[x,y])
                if np.isnan(a) or np.isnan(c): continue
                if a>c*1.05: A_ref+=1
                elif c>a*1.05: A_pred+=1
                else: A_tie+=1
    inside=ndimage.binary_erosion(r)
    shell=ndimage.binary_dilation(r,iterations=2)&~r
    mi,mo=im[inside].mean(),im[shell].mean()
    err=p^r
    closer_in=np.abs(im-mi)<np.abs(im-mo)
    B_ok+=int((err & (closer_in==r)).sum()); B_bad+=int((err & (closer_in!=r)).sum())
print(f'A edge strength at wrong column extremes: ref edge stronger {A_ref}, pred edge stronger {A_pred}, tie {A_tie}  -> P(ref)={A_ref/(A_ref+A_pred):.3f}')
print('   where is the max |gradient| among offsets -2..+2 of the reference edge:',np.round(rank_hits/rank_n,3))
print(f'B region likeness of error voxels: intensity agrees with reference label {B_ok}, with prediction {B_bad} -> {B_ok/(B_ok+B_bad):.3f}')
