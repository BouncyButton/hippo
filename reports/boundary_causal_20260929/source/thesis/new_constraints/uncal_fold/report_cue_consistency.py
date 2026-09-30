"""Build reviewable reports and selected native-slice evidence for cue audit."""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR','/tmp/hippo-uncal-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .audit_validation_profiles import ROOT, load_case, panel

OUT=ROOT/'experiments/uncal_cue_consistency_20260921'
DOC=ROOT/'docs/experiments/uncal_cue_consistency_20260921'


def pct(v): return f'{100*v:.1f}%'


def evidence(case, record, models, selected, out):
    c=case['cut']; mask=case['labels']>0
    # Select review planes from the union only; selection does not verify anatomy.
    def gap_count(column):
        edges=np.diff(np.pad(column.astype(int),1))
        return max(0,int((edges==1).sum())-1)
    xs=np.array([gap_count(column) for column in mask[:,c,:]])
    zs=np.array([gap_count(column) for column in mask[:,c,:].T])
    x=int(np.argmax(xs)) if xs.max()>0 else int(np.argwhere(mask[:,c,:])[:,0].mean())
    z=int(np.argmax(zs)) if zs.max()>0 else int(np.argwhere(mask[:,c,:])[:,1].mean())
    pred=next(p['pred'] for p in models[selected]['predictions'] if p['name']==case['name'])
    planes=[(1,c+d) for d in (2,1,0,-1,-2)]+[(1,pred),(0,x),(2,z)]
    fig,axs=plt.subplots(2,8,figsize=(17,5.6))
    for col,(axis,index) in enumerate(planes):
        for overlay in (0,1):
            panel(axs[overlay,col],case,axis,index,bool(overlay))
        suffix=' | cut' if col==2 else (' | predicted' if col==5 else '')
        axs[0,col].set_title(f"{'xyz'[axis]}={index}"+suffix,fontsize=10)
    fig.suptitle(f"{record['split']} {case['name']} | fitted cut y={c}; selected ranker y={pred}\n"
                 "Raw MRI above; label outlines below. Blue anterior, orange posterior. Native indices.\n"
                 "Sagittal/axial planes selected by gap count, otherwise centroid; no verified apex marker.",fontsize=11)
    fig.tight_layout(rect=(0,0,1,.86))
    fig.savefig(out/f"{case['name']}.png",dpi=140);plt.close(fig)


def main():
    DOC.mkdir(parents=True,exist_ok=True)
    s=json.loads((OUT/'summary.json').read_text());cases=json.loads((OUT/'features.json').read_text())
    # Also evaluate discrimination against the six immediate neighbours, so
    # a broad head/body difference cannot masquerade as precise localisation.
    for group in ('train','val'):
        local_scores=[]
        for c in (c for c in cases if c['split']==group):
            y=np.array(c['candidates']);x=np.array(c['X']);t=c['target']
            at=x[y==t][0];local=x[(abs(y-t)<=3)&(y!=t)]
            local_scores.append(np.mean((at>local)+.5*(at==local),axis=0))
        mean_local=np.mean(local_scores,axis=0)
        for r in s['scalars']:
            value=float(mean_local[r['index']])
            r[group]['local_auc']=value if r['direction']==1 else 1-value
    (OUT/'summary.json').write_text(json.dumps(s,indent=2)+'\n')
    models=s['models']; selected=min(models,key=lambda n:models[n]['train_oof']['mae'])
    relative='../../../experiments/uncal_cue_consistency_20260921/'
    lines=['# Broad MRI/shape cue audit: fold 0 training and validation','',
           '2026-09-21. Exploratory analysis of 208 training and 52 validation **volumes**. Case-level splits cannot be assumed to separate subjects: the released crops lack a verified left/right subject-pair mapping. Both sets have informed earlier research, so this is a consistency audit, not a fresh holdout experiment.','',
           '[Read the interpretation and LTN implications](interpretation.md).','',
           '## What was tested','',
           f"{s['n_features']} scalar measurements cover separate profiles, disappearance of small profiles, vertical/horizontal tissue runs, holes, superior notches and protrusions, adjacent-slice area/contour changes, sagittal/axial anterior bridges, and local T1 intensity/edge/context changes. Current values and adjacent-slice differences are tested. This is a broad explicit candidate set, not every possible anatomical cue.",'',
           'The sagittal return proxy finds vertically separated tissue runs at a candidate y and requires the gap interval to be bridged at the same x within 1–4 mm anteriorly. The analogous axial proxy works along x at a fixed z. Minimum tissue-run lengths of one and two voxels test sensitivity to thin profiles. These axis-aligned bridge tests can miss curved folds and can flag unrelated clefts. MRI darkness inside these gaps is also measured; no CSF or named structure is segmented.','',
           'Features receive MRI and the union of anterior/posterior reference masks, never the class identity. This is oracle foreground support. Targets are fitted class interfaces; they are not independent anatomical apex annotations. Raw MRI is robustly scaled per volume. Candidate cuts cover all occupied native coronal planes except the most posterior one, because a cut requires tissue on both sides. Increasing y is anterior; all coordinates are native zero-based indices.','',
           '## Discrete cue coverage','',
           'Counts are volumes, not slices. “At cut” is the fitted interface. “Near” means within 2 mm. “Elsewhere” means the cue also occurs more than 3 mm from the interface. Near-cut coverage alone is not localisation accuracy.','',
           '| Cue | Train at cut /208 | Val at cut /52 | Train near /208 | Val near /52 | Train elsewhere /208 | Val elsewhere /52 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for cue,t in s['events']['train'].items():
        v=s['events']['val'][cue]
        lines.append(f"| {cue} | {t['at_cut']} | {v['at_cut']} | {t['near2']} | {v['near2']} | {t['far']} | {v['far']} |")
    lines+=['','Two components require eight-neighbour separation and at least two voxels in each component. The disappearance flag requires a secondary component to have no same-coordinate overlap with the immediately posterior mask; shape displacement can also produce this flag. Neither condition establishes anatomical identity.','',
            '## Strongest individual measurements on training data','',
            'Direction and threshold are chosen using training only. Ranking below uses training within-volume AUC: how often the cut scores above a distant candidate, with ties worth 0.5. Thresholds maximise training sensitivity minus mean within-volume distant-slice false-positive rate over 101 quantiles. Training values are fitted/descriptive and optimistic; validation values are not used to choose direction, threshold, or table order. Multiple candidates were explored; no confirmatory significance claim is made.','',
            '| Measurement | Direction | Train distant AUC | Val distant AUC | Val local AUC | Train cut sensitivity | Val cut sensitivity | Val distant-slice flag rate |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
    strongest=sorted(s['scalars'],key=lambda r:r['train']['mean_within_case_auc'],reverse=True)
    for r in strongest[:15]:
        t,v=r['train'],r['val']
        lines.append(f"| {r['name']} | {r['direction']:+d} | {t['mean_within_case_auc']:.3f} | {v['mean_within_case_auc']:.3f} | {v['local_auc']:.3f} | {pct(t['cut_detection'])} | {pct(v['cut_detection'])} | {pct(v['far_slice_flag_rate'])} |")
    lines+=['','Local AUC compares the cut with the other candidates within 3 mm, using the same training-fitted score direction. A value near 0.5 means the scalar does not distinguish the exact cut from its neighbours, even if it separates this broad region from distant slices.','']
    lines+=['','## Can combinations locate the cut?','',
            'A fixed logistic ranker (C=1) combines each feature family, with and without a relative-position prior. Four-fold case-wise out-of-fold predictions are computed inside the 208 training volumes; scalers and position statistics are fitted separately within each fold. One final fit on all 208 predicts validation. Each training case has equal total weight and equal positive/negative mass. The highest score selects the cut; exact ties use the middle tied candidate. No segmentation network was trained.','',
            '| Inputs | Train OOF MAE, mm | Val MAE, mm | Val exact | Val within 1 mm | Val within 2 mm | Val maximum, mm |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for name,m in models.items():
        t,v=m['train_oof'],m['val']
        lines.append(f"| {name} | {t['mae']:.3f} | {v['mae']:.3f} | {pct(v['exact'])} | {pct(v['within1'])} | {pct(v['within2'])} | {v['maximum']} |")
    lines+=['',f'Lowest training OOF MAE selects **{selected}**. The comparison against position-only is required: a cue that merely recognises the usual location is not enough evidence for an anatomical landmark detector.','']
    # Descriptive paired bootstrap; resampling units are volumes, not subjects.
    base={p['name']:abs(p['pred']-p['target']) for p in models['position_only']['predictions']}
    pred=models[selected]['predictions'];val=[p for p in pred if p['split']=='val']
    delta=np.array([base[p['name']]-abs(p['pred']-p['target']) for p in val])
    rng=np.random.default_rng(0);boot=delta[rng.integers(0,len(delta),(5000,len(delta)))].mean(1)
    interval=np.quantile(boot,[.025,.975])
    lines += [f"Validation MAE reduction versus position-only: {delta.mean():.3f} mm; descriptive volume-bootstrap 95% interval [{interval[0]:.3f}, {interval[1]:.3f}] mm. Better/equal/worse volumes: {int((delta>0).sum())}/{int((delta==0).sum())}/{int((delta<0).sum())}. This interval does not account for model selection, prior validation exploration, or unknown paired hippocampi.",'']
    for group in ('train','val'):
        sub=[c for c in cases if c['split']==group];pmap={p['name']:p['pred'] for p in pred}
        for planar in (True,False):
            subset=[c for c in sub if (c['plane_disagreement']==0)==planar]
            errors=[abs(pmap[c['name']]-c['target']) for c in subset]
            if errors:lines.append(f"- {group}, {'planar' if planar else 'nonplanar'} reference split: n={len(subset)}, selected-model MAE {np.mean(errors):.3f} mm.")
    actual=np.array([abs(pmap[c['name']]-c['actual_last_anterior']) for c in cases if c['split']=='val'])
    lines+=['',f"Using the actual most-posterior anterior voxel plane instead of the fitted plane gives selected-model validation MAE {actual.mean():.3f} mm.",'']
    if (OUT/'sensitivity.json').exists():
        sensitivity=json.loads((OUT/'sensitivity.json').read_text())
        lines+=['## Post-hoc diagnostic sensitivity','',
                'Inspection of the original largest validation error (033, predicted y=6 versus target y=26) showed that the closing-based solidity proxy is nearly constant in training but extreme at that posterior tip. Its three standardised terms dominate the score. This is a descriptor/extrapolation failure, not evidence of absent uncal anatomy.','',
                f"Removing only the current/delta/next-delta solidity proxy and retaining the same four training folds, C=1 and other inputs yields training OOF MAE **{sensitivity['train_oof']['mae']:.3f} mm** and validation MAE **{sensitivity['val']['mae']:.3f} mm**, within 1 mm **{pct(sensitivity['val']['within1'])}**, maximum **{sensitivity['val']['maximum']} mm**. The position-only comparator remains 1.159/1.173 mm on training OOF/validation.",'',
                'This exclusion was motivated by a validation failure. Its improved result is explicitly exploratory and requires independent evaluation. Original results and original-model gallery predictions remain above; they have not been replaced by the corrected variant.',
                f'[Sensitivity results]({relative}sensitivity.json) and [033 score contributions]({relative}case033_failure.json).','']
    lines+=['## Evidence gallery and every-case results','']
    # Include known validation examples plus training cue-positive and negative controls.
    selected_names=['hippocampus_185','hippocampus_205','hippocampus_164','hippocampus_327']
    reasons={n:'Previously discussed validation example' for n in selected_names}
    for cue in ('secondary_disappears','sagittal_return_min1','axial_return_min2'):
        positive=[r['name'] for r in s['events']['train'][cue]['cases'] if r['at_cut']]
        for n in positive[:2]:
            if n not in reasons:selected_names.append(n);reasons[n]=f'Training positive: {cue}'
    training_pred=[p for p in pred if p['split']=='train']
    for p in sorted(training_pred,key=lambda p:abs(p['pred']-p['target']),reverse=True)[:2]:
        if p['name'] not in reasons:selected_names.append(p['name']);reasons[p['name']]='Largest training OOF localisation errors'
    for p in sorted(val,key=lambda p:abs(p['pred']-p['target']),reverse=True)[:2]:
        if p['name'] not in reasons:selected_names.append(p['name']);reasons[p['name']]='Largest validation localisation errors'
    records={c['name']:c for c in cases}
    for n in selected_names:
        evidence(load_case(ROOT/'datasets/Dataset101_MSD',n),records[n],models,selected,OUT)
        lines.append(f'- [{n}]({relative}{n}.png): {reasons[n]}.')
    lines+=['','These examples are selected for evidence inspection, not sampled to estimate expert visibility. Automated extraction covers every volume; detailed manual review does not.','',
            f'[All 260 case results](cases.md), [full numerical results]({relative}summary.json), and [summary plot]({relative}summary.png).','',
            '## Boundaries on the conclusion','',
            '- Disconnected components, double runs, or bridges are geometric cues. Their absence cannot establish absence of the uncus; their presence cannot establish uncal identity.',
            '- Reliable hemisphere/medial direction is not supplied by the crop metadata. Side descriptors are symmetrised; no medial-versus-lateral anatomical claim is made.',
            '- Surrounding named structures such as crus cerebri and ventricular recess are not independently labelled. Local context intensity is only a proxy; the complete Woolard procedure is not automated here.',
            '- Reference union masks encode expert delineation information. Results must be repeated on predicted foreground before using a deployable MRI/shape predicate.',
            '- Cue/annotation agreement does not establish generalisation gains from an LTN loss. Independent landmarks and a new untouched evaluation split are needed for that claim.',
            '- No labels, baseline models, or training losses were modified.','',
            '## Reproduction','',
            '```bash',
            'rtk proxy .venv/bin/python -m thesis.new_constraints.uncal_fold.audit_cue_consistency',
            'rtk proxy .venv/bin/python -m thesis.new_constraints.uncal_fold.audit_cue_sensitivity',
            'rtk proxy .venv/bin/python -m thesis.new_constraints.uncal_fold.report_cue_consistency',
            'rtk proxy .venv/bin/python -m pytest thesis/new_constraints/uncal_fold/test_cue_consistency.py -q',
            '```','',f"Split SHA256: `{s['split_sha256']}`."]
    (DOC/'README.md').write_text('\n'.join(lines)+'\n')
    # Detailed tabular record for every volume.
    lines=['# Every-case cue results','',f'Selected ranker: {selected}. Training predictions are out of fold. Validation predictions use the final training fit. Cues refer to the fitted cut. No row is an independent anatomical annotation.','',
           '| Case | Split | Fitted cut | Actual last anterior | Plane disagreement | Predicted cut | Error | Discrete cues at cut |',
           '|---|---|---:|---:|---:|---:|---:|---|']
    pmap={p['name']:p['pred'] for p in pred}
    for c in cases:
        cues=[]
        for cue,event in s['events'][c['split']].items():
            if next(r for r in event['cases'] if r['name']==c['name'])['at_cut']:cues.append(cue)
        name=c['name'];label=f'[{name}]({relative}{name}.png)' if name in selected_names else name
        lines.append(f"| {label} | {c['split']} | {c['target']} | {c['actual_last_anterior']} | {c['plane_disagreement']} | {pmap[name]} | {abs(pmap[name]-c['target'])} | {', '.join(cues) or 'None of these discrete cues'} |")
    (DOC/'cases.md').write_text('\n'.join(lines)+'\n')
    (OUT/'gallery_selection.json').write_text(json.dumps(reasons,indent=2)+'\n')
    fig,axs=plt.subplots(1,2,figsize=(14,6))
    keys=list(s['events']['train']); y=np.arange(len(keys))
    for offset,group,total,col in [(-.18,'train',208,'#2285bd'),(.18,'val',52,'#ee9029')]:
        axs[0].barh(y+offset,[100*s['events'][group][k]['at_cut']/total for k in keys],height=.34,color=col,label=group)
    axs[0].set_yticks(y,keys);axs[0].set_xlabel('Volumes with cue at the fitted cut (%)');axs[0].legend();axs[0].invert_yaxis()
    keys=list(models);y=np.arange(len(keys))
    for offset,group,col in [(-.18,'train_oof','#2285bd'),(.18,'val','#ee9029')]:
        axs[1].barh(y+offset,[models[k][group]['mae'] for k in keys],height=.34,color=col,label=group)
    axs[1].set_yticks(y,keys);axs[1].set_xlabel('Cut localisation MAE (native mm)');axs[1].legend();axs[1].invert_yaxis()
    fig.suptitle('Exploratory cue consistency: 208 training / 52 validation volumes\nReference foreground; annotation agreement, not anatomical validation',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.91));fig.savefig(OUT/'summary.png',dpi=150);plt.close(fig)
    print('Report ready. Selected model:',selected,'Gallery:',len(selected_names))


if __name__=='__main__':main()
