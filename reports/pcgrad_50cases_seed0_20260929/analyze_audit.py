"""Summarize saved case-level audits; no training or model selection."""
import hashlib
import json
import statistics as st
from pathlib import Path

OUT = Path(__file__).resolve().parent
OLD = OUT.parent/'pcgrad_edge_20260929/results'
NEW = OUT/'results'
EXTRA = OUT/'supplement/results'

def read(p):
    return json.loads(p.read_text())

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def save(p, x):
    p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def means(rows):
    return {k:st.mean(r['metrics'][k] for r in rows) for k in rows[0]['metrics']}

def enrich(r):
    m = dict(r['metrics'])
    for side in ('inner','outer'):
        m[side+'_agreement'] = 1-m[side+'_disagree']
    m['balanced_edge_soft_mse'] = st.mean(m[k+'_soft_squared_error'] for k in ('inner','outer','cross'))
    tp = m['gt_foreground_voxels']-m['all_fn']
    m['foreground_precision'] = tp/max(m['predicted_foreground_voxels'],1)
    m['foreground_recall'] = tp/max(m['gt_foreground_voxels'],1)
    m['predicted_over_gt_volume'] = m['predicted_foreground_voxels']/max(m['gt_foreground_voxels'],1)
    m['exactly_one_component'] = int(m['foreground_components_6']==1)
    return dict(split=r['split'],case_name=r['case_name'],metrics=m)

def table(lines, title, metrics, columns, groups):
    lines.extend(['', '## '+title, '', '| Metric | '+' | '.join(columns)+' |',
                  '|---|'+'---:|'*len(columns)])
    for label,key,scale in metrics:
        lines.append('| '+label+' | '+' | '.join(f'{g[key]*scale:.4f}' for g in groups)+' |')

def main():
    v = read(EXTRA/'VERIFICATION.json')
    assert v['status']=='complete' and v['training'] is False
    for name, expected in v['outputs'].items():
        assert sha(EXTRA/name)==expected
    assert read(NEW/'LAUNCHER_EXIT.json')['exit_code']==0
    runs, case_rows = {}, []
    old_names = None
    for label,base,arm in [('sum10',OLD,'sum'),('pcgrad10',OLD,'pcgrad'),('pcgrad50',NEW,'pcgrad')]:
        p = base/'fold0/seed0'/arm
        done = read(p/'completion.json')
        assert sha(p/'selected_cases.json')==done['selected_cases_sha256']
        assert v['runs'][label]['checkpoint_sha256']==done['checkpoint_sha256']
        extra = {(r['split'],r['case_name']):r for r in read(EXTRA/(label+'.json'))['cases']}
        rows=[]
        for r in read(p/'selected_cases.json'):
            row=enrich(r)
            supplement=extra[(r['split'],r['case_name'])]
            row['metrics'].update({'bands_'+k:float(x) for k,x in supplement['bands'].items()})
            row['metrics'].update(supplement['surface'])
            rows.append(row)
            case_rows.append(dict(run=label,**row))
        tr=[r for r in rows if r['split']=='train']
        va=[r for r in rows if r['split']=='validation']
        assert (len(tr),len(va))==((50,52) if label=='pcgrad50' else (10,52))
        names={r['case_name'] for r in tr}
        if old_names is None: old_names=names
        assert old_names<=names
        train, val = means(tr), means(va)
        runs[label]=dict(selected_epoch=done['selected_epoch'],stopped_epoch=done['stopped_epoch'],
            selected_updates=done['selected_epoch']*len(tr),final_updates=done['stopped_epoch']*len(tr),
            train=train,validation=val,gap={k:train[k]-val[k] for k in train},
            train_original10=means([r for r in tr if r['case_name'] in old_names]))
        if label=='pcgrad50':
            runs[label]['train_added40']=means([r for r in tr if r['case_name'] not in old_names])
    keys=['macro_dice','union_dice','inner_both_correct','outer_both_correct','cross_correct_transition',
          'all_fp','all_fn','bands_case_loss','symmetric_mean_surface_distance_voxels',
          'foreground_components_6']
    paired={}
    for ref in ('pcgrad10','sum10'):
        a={r['case_name']:r['metrics'] for r in case_rows if r['run']==ref and r['split']=='validation'}
        b={r['case_name']:r['metrics'] for r in case_rows if r['run']=='pcgrad50' and r['split']=='validation'}
        assert a.keys()==b.keys() and len(a)==52
        paired[ref]={}
        for k in keys:
            delta=[b[n][k]-a[n][k] for n in sorted(a)]
            sign=-1 if k in ('all_fp','all_fn','bands_case_loss','symmetric_mean_surface_distance_voxels','foreground_components_6') else 1
            paired[ref][k]=dict(mean_delta=st.mean(delta),median_delta=st.median(delta),
                improved=sum(sign*d>1e-12 for d in delta),worsened=sum(sign*d < -1e-12 for d in delta),
                unchanged=sum(abs(d)<=1e-12 for d in delta),min_delta=min(delta),max_delta=max(delta))
    p=NEW/'fold0/seed0/pcgrad'
    temporal=[]
    for f in sorted((p/'coherence').glob('*.json')):
        rows=[enrich(r) for r in read(f)]
        epoch=int(f.stem.split('_')[-1])
        tr=means([r for r in rows if r['split']=='train'])
        va=means([r for r in rows if r['split']=='validation'])
        temporal.append(dict(epoch=epoch,train=tr,validation=va,gap={k:tr[k]-va[k] for k in tr}))
    history=sorted([read(f) for f in (p/'epochs').glob('*.json')],key=lambda r:r['epoch'])
    update_files=[read(f) for f in (p/'updates').glob('*.json')]
    update_windows=[]
    for lo,hi in ((1,15),(16,19),(20,30),(31,60)):
        q=[r for e in update_files if lo<=e['epoch']<=hi for r in e['steps']]
        update_windows.append(dict(epochs=[lo,hi],updates=len(q),skipped=sum(r['optimizer_step_skipped'] for r in q),
            projection_fraction=st.mean(r['projection_count']>0 for r in q),
            median_projected_over_original_norm=st.median(r['projected_over_original_norm'] for r in q),
            median_cosine_with_original=st.median(r['cosine_with_original'] for r in q)))
    assert sum(w['updates'] for w in update_windows)==3000
    result=dict(scope='Fold 0 seed 0 only; selected checkpoints; same 52 validation cases.',
        averaging='Equal case means; train original10 and added40 shown separately. No independent treatment of repeated epochs/cases.',
        selected=runs,paired_validation=paired,temporal=temporal,history=history,update_windows=update_windows,
        inference_verification=v,limitations=['No 50-case summed-loss control','One seed and one validation fold',
            'Data, update budget, LR schedule and update-count warmup changed together',
            'Validation was used for checkpoint selection and LR scheduling; no independent test set'])
    save(OUT/'AUDIT_SUMMARY.json',result)
    save(OUT/'AUDIT_CASES.json',case_rows)
    save(OUT/'AUDIT_TEMPORAL_CASES.json',[dict(epoch=int(f.stem.split('_')[-1]),**r)
        for f in sorted((p/'coherence').glob('*.json')) for r in read(f)])
    save(OUT/'AUDIT_INPUT_HASHES.json',{str(f.relative_to(OUT)):sha(f)
        for folder in (NEW,OUT/'supplement') for f in folder.rglob('*') if f.is_file()})
    lines=['# Seed 0 PCGrad, 50 training cases: anatomical and constraint audit','',
        '**The 50-case solution is substantially more anatomically plausible than the earlier 10-case PCGrad solution, and outperforms the earlier 10-case summed-loss model on this validation fold. It does not improve every constraint, and substantial train–validation gaps remain.**','',
        'Training job 676272 completed successfully in 16m54s, stopping at epoch 60 (3,000 updates). The unchanged selected checkpoint is epoch 30 (1,500 updates). Supplementary job 676288 performed inference only on three selected checkpoints; no model was trained or modified.','',
        '## Definitions and verification','',
        'Foreground is the union of anterior/posterior classes: y=1[label>0], p=P(class 1)+P(class 2), H=1[argmax of all three classes>0]. H is not obtained by thresholding p at 0.5. Inner and outer bands are the foreground shell removed by two 6-neighbor erosions and the background shell added by two dilations. Face pairs are counted once along positive axes, with both endpoints in these bands. These are local boundary-shell constraints, not global hippocampal connectivity or anterior/posterior anatomical constraints.','',
        '- Inner/outer hard equality satisfaction = mean 1[H_i=H_j] = 1−disagreement. Both-wrong pairs satisfy equality. Pair correctness = mean 1[H_i=y_i and H_j=y_j]; we report it separately.',
        '- Crossing correctness is `correct_transition` = mean 1[H_i−H_j=y_i−y_j] on crossing ground-truth faces. It is not its complement, not Dice, and not 1−MSE.',
        '- Soft edge error = mean ((p_i−p_j)−(y_i−y_j))² per pair group. The separated edge objective gives the three group means equal weight. Low same-side MSE can occur with an incorrectly constant foreground prediction.',
        '- Bands use the original zero-gamma, zero-degree balanced BCE: B=(mean_inner[−log p]+mean_outer[−log(1−p)])/2 (implementation denominator epsilon 1e−6). Existing fuzzy truth is exp(−B) per case, averaged after exponentiation. It is a confidence score, not a percentage of correct voxels or edges. The existing binary case-level flag is exp(−B)≥0.90.',
        '- Hard macro Dice averages anterior/posterior class Dice within each case; union Dice measures the foreground envelope. Logged training `train_dice` is a soft loss measured during updates and is not used as audited hard Dice.',
        '- Connectivity is computed on the predicted foreground with 6-neighbor adjacency. Ground truth is not always a single component, so one component is supportive but not proof of perfect anatomy.',
        '- Supplementary symmetric surface distances concatenate both directional distances between 6-neighbor surface voxels. The mean and 95th percentile are reported in evaluation-grid voxels, not millimetres; no patient-specific spacing claim is made.','',
        'All original selected-case hashes and selected-checkpoint hashes were checked; both frozen source manifests were verified and shared source files are byte-identical. Supplementary inference used the original Torch/MONAI/NumPy versions and CUDA AMP. Original saved edge/Dice values remain primary; newly computed bands and surface measurements are separate. Drift is quantified in AUDIT_SUMMARY.json. Cases are averaged equally; repeated cases or epochs are not independent samples.']
    core=[('Macro Dice (%)','macro_dice',100),('Union Dice (%)','union_dice',100),
          ('Inner pair correctness (%)','inner_both_correct',100),('Outer pair correctness (%)','outer_both_correct',100),
          ('Correct crossing transition (%)','cross_correct_transition',100),
          ('Inner equality satisfaction (%)','inner_agreement',100),('Outer equality satisfaction (%)','outer_agreement',100),
          ('Bands fuzzy truth (0–1)','bands_truth',1),('Bands inner BCE','bands_inner_loss',1),
          ('Bands outer BCE','bands_outer_loss',1),('Bands balanced BCE','bands_case_loss',1),
          ('Bands confidence-adherent cases (%)','bands_confidence_adherent',100),
          ('Inner soft MSE','inner_soft_squared_error',1),('Outer soft MSE','outer_soft_squared_error',1),
          ('Crossing soft MSE','cross_soft_squared_error',1),('Balanced edge soft MSE','balanced_edge_soft_mse',1)]
    n=runs['pcgrad50']
    table(lines,'Selected epoch 30: train and validation',core,['Train (50)','Validation (52)','Train − validation'],[n['train'],n['validation'],n['gap']])
    lines.extend(['','Gaps in percentage rows are percentage points; fuzzy-truth gaps are score units. For BCE/MSE, lower is better, so negative train−validation indicates lower error on training.', '',
        'Bands confidence improves substantially overall, but its inner and outer terms move differently: compared with the original ten-case models, inner BCE worsens while outer BCE improves. The existing confidence-adherent flag is zero for every selected case in all three runs because none reaches the predefined 0.90 threshold. This is a floor effect of a case-level confidence criterion, not zero correctly classified voxels.'])
    anatomy=[('Macro Dice (%)','macro_dice',100),('Union Dice (%)','union_dice',100),
        ('Inner pair correctness (%)','inner_both_correct',100),('Outer pair correctness (%)','outer_both_correct',100),
        ('Correct crossing transition (%)','cross_correct_transition',100),('Bands fuzzy truth','bands_truth',1),
        ('Bands balanced BCE','bands_case_loss',1),('False positives per case','all_fp',1),('False negatives per case','all_fn',1),
        ('Precision (%)','foreground_precision',100),('Recall (%)','foreground_recall',100),
        ('Predicted/GT volume ratio','predicted_over_gt_volume',1),('Components per case','foreground_components_6',1),
        ('Voxels outside largest component','foreground_outside_largest_6',1),
        ('Enclosed background voxels','enclosed_background_voxels_6',1),
        ('Mean surface distance (voxels)','symmetric_mean_surface_distance_voxels',1),
        ('Surface-distance 95th percentile (voxels)','symmetric_surface_distance_p95_voxels',1)]
    table(lines,'Validation: same 52 cases',anatomy,['10-case sum (epoch 64)','10-case PCGrad (epoch 61)','50-case PCGrad (epoch 30)'],[runs[k]['validation'] for k in ('sum10','pcgrad10','pcgrad50')])
    table(lines,'Training comparison on the identical original ten cases',core,['10-case sum','10-case PCGrad','50-case PCGrad: original ten'],[runs[k]['train_original10'] for k in ('sum10','pcgrad10','pcgrad50')])
    table(lines,'Does training improvement extend to the added cases?',core,['Original ten','Added forty'],[n['train_original10'],n['train_added40']])
    table(lines,'Train minus validation gaps (training cohort differs for the 50-case run)',
        [('Macro Dice (pp)','macro_dice',100),('Inner correctness (pp)','inner_both_correct',100),
         ('Outer correctness (pp)','outer_both_correct',100),('Crossing correctness (pp)','cross_correct_transition',100),
         ('Bands fuzzy truth (score units)','bands_truth',1)],
        ['10-case sum','10-case PCGrad','50-case PCGrad'],[runs[k]['gap'] for k in ('sum10','pcgrad10','pcgrad50')])
    lines.extend(['','## Case-level consistency on validation','','| Metric | Versus 10-case PCGrad: improved / worse / tied | Versus 10-case sum: improved / worse / tied |','|---|---:|---:|'])
    for k in keys:
        vals=[paired[a][k] for a in ('pcgrad10','sum10')]
        lines.append('| '+k+' | '+' | '.join(f"{r['improved']} / {r['worsened']} / {r['unchanged']}" for r in vals)+' |')
    lines.extend(['','These are descriptive paired counts, not 52 independent experimental replications. A lower component count is counted here as a directional change toward one; component count alone is not an anatomical ground-truth metric.','',
        '## Evolution during training','','Entries are train / validation percentages. These are full 50/52-case audits, unlike the earlier experiment’s two-case training probes.','',
        '| Epoch | Updates | Macro Dice | Inner correct | Outer correct | Crossing correct |','|---:|---:|---:|---:|---:|---:|'])
    for r in temporal:
        lines.append(f"| {r['epoch']} | {r['epoch']*50} | "+' | '.join(f"{100*r['train'][k]:.2f} / {100*r['validation'][k]:.2f}" for k in ('macro_dice','inner_both_correct','outer_both_correct','cross_correct_transition'))+' |')
    lines.extend(['','At epoch 5, inner correctness is effectively perfect on both splits while outer/crossing correctness is near zero: this is extensive foreground overprediction, not complete anatomical learning. At epoch 15 it persists, with roughly 3,800 false positives per case. The major validation Dice jump occurs at epoch 20 (1,000 updates), from 65.59% at epoch 19 to 82.26%. By epoch 30 the foreground envelope is substantially corrected. Intermediate full anatomical audits are at epochs 15 and 30; they do not locate every boundary-metric change precisely at epoch 20.','',
        'From epoch 30 to 60, train Dice rises from 94.64% to 96.14% and train crossing correctness from 63.54% to 72.31%. Validation Dice changes from 85.35% to 85.17%, and crossing correctness from 40.28% to 39.90%. The crossing gap widens from 23.26 to 32.41 percentage points; the inner gap from 13.08 to 16.23. This is evidence of continued training fit without matching validation improvement. It supports a remaining generalization limitation, especially for exact boundary transitions, alongside ordinary segmentation overfitting. It does not establish a uniquely constraint-caused overfitting mechanism.','',
        '## Optimization and attribution','',
        'All 3,000 optimizer steps completed without skips; projection occurred at every update. Median projected/original gradient norm ratios by windows 1–15, 16–19, 20–30, 31–60 are '+', '.join(f"{w['median_projected_over_original_norm']:.3f}" for w in update_windows)+'. These observations do not isolate the cause of the transition.','',
        'The new selected checkpoint has 1,500 updates versus 610 for the old PCGrad selected checkpoint (690 at its stop). The successful epoch-20 transition occurs after 1,000 updates, beyond the old run’s entire update budget. This is consistent with a data/update-budget limitation, but cannot distinguish it from effects of more diverse cases, the adaptive LR, or the five-epoch warmup now lasting 250 rather than 50 updates. The LR was still 1e−4 when Dice jumped.','',
        '## Interpretation','',
        '1. **Anatomically promising: yes, relative to the measured 10-case solutions.** The model predicts a much more accurate foreground envelope, with fewer false positives, better outer correctness and correct boundary transitions, and no detached foreground islands at the selected checkpoint. Both union and anterior/posterior macro Dice improve. This is evidence of improved segmentation geometry, not direct evidence of a learned internal anatomical representation.',
        '2. **Every constraint learned better: no.** Crossing and outer hard correctness improve on both train and validation, but inner hard correctness decreases and false negatives increase. Same-side probability MSE is not equivalent to correctness: a constant overlarge foreground can have almost zero MSE. The report keeps equality satisfaction, correct labels, soft error and unary bands confidence separate.',
        '3. **Constraints generalize perfectly: no.** The selected model retains meaningful train–validation gaps, especially in crossing correctness, and the gaps grow with later training. Better absolute validation performance coexists with a larger gap than the poorly fitting old PCGrad model.',
        '4. **PCGrad is established as superior: no.** There is no matched 50-case summed-loss control, only one seed, and the reused validation set drove both LR adaptation and selection. More examples, more updates, and LR changes are confounded. The result shows that this PCGrad setup can reach a much better solution under the new regime; it does not identify PCGrad as the source of the advantage.',
        '5. **No further experiment was launched.** The supplementary job only evaluated frozen selected checkpoints; original training outputs and checkpoints remain untouched.','',
        'Machine-readable selected, paired, temporal, optimization and provenance results: AUDIT_SUMMARY.json. Per-case measurements: AUDIT_CASES.json and AUDIT_TEMPORAL_CASES.json.','',
        'Maximum absolute supplementary-inference hard Dice drift across all three checkpoints is '+f"{100*max(r['max_drift']['macro_dice'] for r in v['runs'].values()):.5f}"+' percentage points for an individual case; maximum crossing-correctness drift is '+f"{100*max(r['max_drift']['cross_correct_transition'] for r in v['runs'].values()):.5f}"+' percentage points. These small CUDA inference differences do not approach the measured effects.','',
        f'![Training trajectory]({OUT}/audit_trajectory.png)',''])
    (OUT/'AUDIT_REPORT.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(14,7.5))
    for ax,key,label in zip(axes.flat,('macro_dice','inner_both_correct','outer_both_correct','cross_correct_transition','all_fp','all_fn'),
            ('Macro Dice (%)','Inner pair correctness (%)','Outer pair correctness (%)','Correct crossing transition (%)','False-positive voxels per case','False-negative voxels per case')):
        scale=100 if key not in ('all_fp','all_fn') else 1
        for split,color in [('train','#2269a0'),('validation','#da6a23')]:
            ax.plot([r['epoch'] for r in temporal],[r[split][key]*scale for r in temporal],'-o',label=split,color=color)
        if key=='macro_dice':
            ax.plot([r['epoch'] for r in history],[r['val_dice_hard']*100 for r in history],color='#da6a23',alpha=.4,lw=1)
        ax.axvline(30,ls=':',color='#555555',label='Selected epoch' if key=='macro_dice' else None)
        ax.set_title(label);ax.set_xlabel('Epoch (50 updates per epoch)');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=9)
    fig.suptitle('PCGrad with 50 training cases: improved geometry, remaining generalization gaps')
    fig.tight_layout();fig.savefig(OUT/'audit_trajectory.png',dpi=170);fig.savefig(OUT/'audit_trajectory.pdf');plt.close(fig)
    print(json.dumps(dict(report=str(OUT/'AUDIT_REPORT.md'),cases=len(case_rows),selected=n,paired=paired),indent=2))

if __name__=='__main__':
    main()
