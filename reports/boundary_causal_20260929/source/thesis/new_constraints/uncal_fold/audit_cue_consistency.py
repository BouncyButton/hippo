"""Explore native MRI/union-mask cues on the fixed fold-0 train/val split.

Features never see the anterior/posterior class identity. All targets are
annotation-derived and all foreground support is reference (oracle) support.
These are image/shape proxies, not verified anatomical landmark detectors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from .audit_validation_profiles import ROOT, load_case
from .refined import (extract_image_features, extract_refined_slice_features,
                      extract_transition_features, normalise_volume_intensity)


def runs(column):
    edges = np.diff(np.pad(column.astype(np.int8), 1))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def return_features(union, image, y, min_run=1):
    """Find separated runs that bridge anteriorly in sagittal/axial planes.

    A bridge requires the complete interval between the runs to be occupied
    at the same x (sagittal) or z (axial) within the next four anterior slices.
    The test is intentionally restrictive and does not identify the uncus.
    """
    result = {}
    for axis, plane_name in ((0, 'sagittal'), (2, 'axial')):
        mask = union[:, y, :] if axis == 0 else union[:, y, :].T
        hits, dark, unique = 0, [], set()
        for fixed in np.flatnonzero(mask.any(1)):
            segments = [(a,b) for a,b in runs(mask[fixed]) if b-a >= min_run]
            for (a,b),(c,d) in zip(segments, segments[1:]):
                # Endpoints come from qualifying tissue segments; intervening
                # one-voxel fragments are allowed but not counted as a bridge.
                ahead = union[fixed, y+1:y+5, b-1:c+1] if axis == 0 else union[b-1:c+1, y+1:y+5, fixed].T
                behind = union[fixed, max(0,y-4):y, b-1:c+1] if axis == 0 else union[b-1:c+1, max(0,y-4):y, fixed].T
                if ahead.shape[0] and np.any(ahead.all(1)):
                    hits += 1
                    unique.add(int(fixed))
                    intens = image[fixed,y,:] if axis == 0 else image[:,y,fixed]
                    dark.append(float((intens[b-1]+intens[c])/2-intens[b:c].mean()))
                if ahead.shape[0] and np.any(ahead.all(1)) and not (behind.shape[0] and np.any(behind.all(1))):
                    result[plane_name+'_anterior_only'] = result.get(plane_name+'_anterior_only',0)+1
        result.setdefault(plane_name+'_anterior_only',0)
        result[plane_name+'_bridge_count'] = hits
        result[plane_name+'_bridge_fraction'] = len(unique)/max(int(mask.any(1).sum()),1)
        result[plane_name+'_gap_darkness'] = float(np.mean(dark)) if dark else 0.0
    return result


def topology(mask, previous):
    cc, _ = ndimage.label(mask, np.ones((3,3)))
    sizes = np.bincount(cc.ravel()); sizes[0] = 0
    order = np.argsort(sizes)[::-1]
    meaningful = [i for i in order if sizes[i]>=2]
    secondary = meaningful[1:]
    return {
        'component_count_min2': float(len(meaningful)),
        'secondary_area': float(sum(sizes[i] for i in secondary)),
        'secondary_fraction': float(sum(sizes[i] for i in secondary)/max(mask.sum(),1)),
        'secondary_disappears': float(any(not (previous & (cc==i)).any() for i in secondary)),
        'vertical_double_min1': float(sum(len(runs(c))>=2 for c in mask)),
        'vertical_double_min2': float(sum(sum(b-a>=2 for a,b in runs(c))>=2 for c in mask)),
        'horizontal_double_min2': float(sum(sum(b-a>=2 for a,b in runs(c))>=2 for c in mask.T)),
    }


def extract_case(dataset, name, split):
    c=load_case(dataset,name)
    image=normalise_volume_intensity(c['image']); union=c['labels']>0
    occupied=np.flatnonzero(union.any((0,2))); low,high=int(occupied.min()),int(occupied.max())
    if len(occupied)!=high-low+1: raise ValueError(f'{name}: empty internal slice')
    per_y={}
    for y in occupied:
        m=union[:,y,:]; prev=union[:,max(0,y-1),:]
        vals={'shape__'+k:v for k,v in extract_refined_slice_features(m).items()}
        # Average the two horizontal orientations to avoid guessing hemisphere.
        f=extract_image_features(image[:,y,:],m)
        rev=extract_image_features(image[::-1,y,:],m[::-1])
        vals.update({'image__'+k:(v+rev[k])/2 for k,v in f.items()})
        vals.update({'topology__'+k:v for k,v in topology(m,prev).items()})
        for min_run in (1,2):
            vals.update({f'return{min_run}__'+k:v for k,v in return_features(union,image,int(y),min_run).items()})
        per_y[int(y)]=vals
    base_names=list(per_y[low]); names=[f'{op}__{n}' for op in ('current','delta','next_delta') for n in base_names]
    candidates=list(range(low+1,high+1)); rows=[]
    for y in candidates:
        current=np.array([per_y[y][n] for n in base_names])
        previous=np.array([per_y[y-1][n] for n in base_names])
        following=np.array([per_y[min(y+1,high)][n] for n in base_names])
        trans=extract_transition_features(union[:,y-1,:],union[:,y,:])
        rev=extract_transition_features(union[::-1,y-1,:],union[::-1,y,:])
        rows.append(np.r_[current,current-previous,following-current,[(v+rev[k])/2 for k,v in trans.items()]])
    names += ['transition__'+k for k in trans]
    rows=np.asarray(rows)
    assert np.isfinite(rows).all() and c['cut'] in candidates
    return dict(name=name,split=split,low=low,high=high,target=c['cut'],
                actual_last_anterior=int(np.where(c['labels']==1)[1].min()),
                plane_disagreement=c['cost'],candidates=candidates,names=names,X=rows.tolist())


def choose(scores, candidates):
    # Deterministic central tie-break, independent of the target.
    ids=np.flatnonzero(np.isclose(scores,np.max(scores),rtol=1e-10,atol=1e-12))
    return int(candidates[ids[len(ids)//2]])


def metrics(cases,pred):
    error=np.array([abs(p-c['target']) for c,p in zip(cases,pred)])
    return dict(n=len(cases),mae=float(error.mean()),median=float(np.median(error)),
                within1=float(np.mean(error<=1)),within2=float(np.mean(error<=2)),
                exact=float(np.mean(error==0)),p90=float(np.quantile(error,.9)),maximum=int(error.max()))


def scalar_stats(cases,j,direction,threshold):
    aucs=[]; local_aucs=[]; exact=[]; near=[]; far_rates=[]; far_cases=[]; predictions=[]; contrasts=[]
    for c in cases:
        y=np.array(c['candidates']); v=np.array(c['X'])[:,j]*direction; t=c['target']
        at=v[y==t][0]; far=v[abs(y-t)>3]
        aucs.append(float(np.mean((at>far)+.5*(at==far))))
        local=v[(abs(y-t)<=3)&(y!=t)]
        local_aucs.append(float(np.mean((at>local)+.5*(at==local))))
        contrasts.append(float(at-np.median(far)))
        flags=v>threshold
        exact.append(bool(flags[y==t][0])); near.append(bool(flags[abs(y-t)<=2].any()))
        far_rates.append(float(flags[abs(y-t)>3].mean()));far_cases.append(bool(flags[abs(y-t)>3].any()))
        predictions.append(choose(v,y))
    return dict(mean_within_case_auc=float(np.mean(aucs)),local_auc=float(np.mean(local_aucs)),cut_detection=float(np.mean(exact)),
                near2_detection=float(np.mean(near)),far_slice_flag_rate=float(np.mean(far_rates)),
                far_case_flag_rate=float(np.mean(far_cases)),
                positive_contrast_fraction=float(np.mean(np.array(contrasts)>0)),
                ranking=metrics(cases,predictions))


def fit_scalar(train,j):
    # Direction and threshold use training volumes only. Each case contributes
    # equal mass to the cut and distant-slice populations.
    at=[];far=[];weights=[];aucs=[]
    for c in train:
        y=np.array(c['candidates']);v=np.array(c['X'])[:,j]
        a=v[y==c['target']][0];f=v[abs(y-c['target'])>3]
        at.append(a);far.extend(f);weights.extend(np.full(len(f),1/len(f)))
        aucs.append(np.mean((a>f)+.5*(a==f)))
    direction=1 if np.mean(aucs)>=.5 else -1
    at=np.array(at)*direction;far=np.array(far)*direction;weights=np.array(weights)
    thresholds=np.unique(np.quantile(np.r_[at,far],np.linspace(0,1,101)))
    score=[np.mean(at>t)-np.average(far>t,weights=weights) for t in thresholds]
    return direction,float(thresholds[int(np.argmax(score))])


def design(c,cols,position,median):
    blocks=[]
    if len(cols): blocks.append(np.array(c['X'])[:,cols])
    if position:
        r=(np.array(c['candidates'])-c['low'])/(c['high']-c['low'])
        blocks.append(np.c_[r,r*r,-((r-median)/.075)**2])
    return np.column_stack(blocks)


def fit_predict(train,test,cols,position):
    med=float(np.median([(c['target']-c['low'])/(c['high']-c['low']) for c in train]))
    xs=[]; ys=[]; ws=[]
    for c in train:
        x=design(c,cols,position,med); y=(np.array(c['candidates'])==c['target']).astype(int)
        xs.append(x);ys.append(y);ws.append(np.where(y,.5,.5/(len(y)-1)))
    scaler=StandardScaler().fit(np.vstack(xs))
    model=LogisticRegression(C=1.0,solver='lbfgs',max_iter=2000)
    model.fit(scaler.transform(np.vstack(xs)),np.concatenate(ys),sample_weight=np.concatenate(ws))
    pred=[choose(model.decision_function(scaler.transform(design(c,cols,position,med))),np.array(c['candidates'])) for c in test]
    return pred,dict(median_position=med,iterations=int(model.n_iter_[0]),
                     feature_indices=list(map(int,cols)),coefficients=model.coef_[0].tolist())


def event_stats(cases):
    names=cases[0]['names']
    definitions={
        'two_components':('current__topology__component_count_min2',1.5),
        'secondary_disappears':('current__topology__secondary_disappears',.5),
        'vertical_double_min2':('current__topology__vertical_double_min2',1.5),
        'sagittal_return_min1':('current__return1__sagittal_bridge_count',.5),
        'sagittal_return_min2':('current__return2__sagittal_bridge_count',.5),
        'axial_return_min1':('current__return1__axial_bridge_count',.5),
        'axial_return_min2':('current__return2__axial_bridge_count',.5),
    }
    result={}
    for event,(name,threshold) in definitions.items():
        j=names.index(name); records=[]
        for c in cases:
            y=np.array(c['candidates']); flags=np.array(c['X'])[:,j]>threshold;t=c['target']
            records.append(dict(name=c['name'],at_cut=bool(flags[y==t][0]),
                                at_actual_last_anterior=bool(flags[y==c['actual_last_anterior']].any()),
                                near2=bool(flags[abs(y-t)<=2].any()),
                                far=bool(flags[abs(y-t)>3].any()),ys=y[flags].tolist()))
        result[event]={k:sum(r[k] for r in records) for k in ('at_cut','at_actual_last_anterior','near2','far')}
        result[event]['cases']=records
    return result


def analyze(cases,out):
    train=[c for c in cases if c['split']=='train'];val=[c for c in cases if c['split']=='val']
    names=cases[0]['names']; scalar=[]
    for j,name in enumerate(names):
        direction,threshold=fit_scalar(train,j)
        scalar.append(dict(name=name,index=j,direction=direction,threshold=threshold,
                           train=scalar_stats(train,j,direction,threshold),
                           val=scalar_stats(val,j,direction,threshold)))
    groups={
        'geometry':[i for i,n in enumerate(names) if '__image__' not in n and 'gap_darkness' not in n],
        'mri':[i for i,n in enumerate(names) if '__image__' in n or 'gap_darkness' in n],
        'crossplane':[i for i,n in enumerate(names) if '__return' in n],
        'all':list(range(len(names))),
    }
    settings=[('position_only',[],True)]
    for group,cols in groups.items():
        settings.extend([(group,cols,False),(group+'_position',cols,True)])
    models={}
    for model_name,cols,position in settings:
        oof=np.zeros(len(train),int)
        for tr,te in KFold(4,shuffle=True,random_state=0).split(train):
            pred,_=fit_predict([train[i] for i in tr],[train[i] for i in te],cols,position)
            oof[te]=pred
        pred,details=fit_predict(train,val,cols,position)
        models[model_name]=dict(train_oof=metrics(train,oof),val=metrics(val,pred),fit=details,
             predictions=[dict(name=c['name'],split=c['split'],target=c['target'],pred=int(p)) for c,p in zip(train+val,list(oof)+pred)])
        print(model_name,models[model_name]['val'],flush=True)
    summary=dict(n_train=len(train),n_val=len(val),n_features=len(names),
                 split_sha256=hashlib.sha256((ROOT/'datasets/Dataset101_MSD/splits_final.json').read_bytes()).hexdigest(),
                 events={s:event_stats(cs) for s,cs in [('train',train),('val',val)]},
                 scalars=scalar,models=models,
                 interpretation='Exploratory case-level audit using reference union masks; not anatomical identification or independent generalisation evidence.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'experiments/uncal_cue_consistency_20260921')
    parser.add_argument('--reuse-features',action='store_true')
    args=parser.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    cache=out/'features.json'
    if args.reuse_features:
        cases=json.loads(cache.read_text())
    else:
        dataset=ROOT/'datasets/Dataset101_MSD';split=json.loads((dataset/'splits_final.json').read_text())[0]
        assert not set(split['train']) & set(split['val'])
        cases=[]
        for group in ('train','val'):
            for name in sorted(split[group]):
                cases.append(extract_case(dataset,name,group))
                if len(cases)%20==0: print('Extracted',len(cases),flush=True)
        cache.write_text(json.dumps(cases)+'\n')
    analyze(cases,out)


if __name__=='__main__':main()
