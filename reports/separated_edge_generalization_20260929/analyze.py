"""Summarize immutable saved audits and the separately recomputed bands audit."""
import hashlib
import json
import statistics as st
import struct
from pathlib import Path

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
BASE=ROOT/'reports/separated_edge_20260929/results'
ARMS=['pooled','separated']

def read(p): return json.loads(p.read_text())
def save(p,x): p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def mean(x): return st.mean(x)
def table(header,rows):
    return '\n'.join(['| '+' | '.join(header)+' |','| '+' | '.join(['---']*len(header))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])

def main():
    manifest=read(BASE/'RESULT_MANIFEST.json')['files']
    for name,digest in manifest.items(): assert sha(BASE/name)==digest,name
    cuda=read(OUT/'CUDA_COMPLETION.json')
    assert cuda['status']=='complete' and cuda['training'] is False and len(cuda['files'])==6
    for name,digest in cuda['files'].items(): assert sha(OUT/name)==digest,name
    soft_reproduction={k:[] for k in ['inner_soft_squared_error','outer_soft_squared_error','cross_soft_squared_error']}
    cases=[]; provenance=[]; temporal_cases=[]; temporal=[]
    for arm in ARMS:
        for seed in range(3):
            p=BASE/f'fold0/seed{seed}/{arm}'; c=read(p/'completion.json')
            assert sha(p/'selected_cases.json')==c['selected_cases_sha256']
            original=read(p/'selected_cases.json')
            bands=read(OUT/f'bands_{arm}_seed{seed}.json')
            assert bands['checkpoint_sha256']==c['checkpoint_sha256']
            lookup={(r['split'],r['case_name']):r for r in bands['cases']}
            assert len(lookup)==62
            provenance.append(dict(arm=arm,seed=seed,selected_epoch=c['selected_epoch'],stopped_epoch=c['stopped_epoch'],checkpoint_sha256=c['checkpoint_sha256'],selected_cases_sha256=c['selected_cases_sha256']))
            for r in original:
                m=dict(r['metrics']); b=lookup[(r['split'],r['case_name'])]
                for side in ['inner','outer']: m[side+'_hard_agreement']=1-m[side+'_disagree']
                for k in soft_reproduction: soft_reproduction[k].append(abs(b['recomputed_edge_metrics'][k]-m[k]))
                for k,v in b['bands'].items(): m['bands_'+k]=float(v)
                cases.append(dict(arm=arm,seed=seed,selected_epoch=c['selected_epoch'],split=r['split'],case_name=r['case_name'],metrics=m,bands_recomputed_drift=b['drift_from_saved']))
            for path in sorted((p/'coherence').glob('epoch_*.json')):
                ep=int(path.stem.split('_')[-1]); validation=read(path); probe=read(p/f'gradients/epoch_{ep:03d}.json')
                for r in validation: temporal_cases.append(dict(arm=arm,seed=seed,epoch=ep,split='validation',case_name=r['case_name'],metrics=r['metrics']))
                for r in probe['cases']: temporal_cases.append(dict(arm=arm,seed=seed,epoch=ep,split='train_probe_2_cases',case_name=r['case_name'],metrics={k+'_soft_squared_error':r['losses'][k] for k in ['inner','outer','cross']},weighted_bands_loss=r['losses']['bands']))
                train_names=[r['case_name'] for r in probe['cases']]
                temporal.append(dict(arm=arm,seed=seed,epoch=ep,train_probe_case_names=train_names,train_probe_n=2,validation_n=52,train_probe={k:mean([r['losses'][k] for r in probe['cases']]) for k in ['inner','outer','cross']},validation={k:mean([r['metrics'][k] for r in validation]) for k in validation[0]['metrics']}))
    cohorts={}
    for arm in ARMS:
        for seed in range(3):
            for split,n in [('train',10),('validation',52)]:
                names={r['case_name'] for r in cases if r['arm']==arm and r['seed']==seed and r['split']==split}
                assert len(names)==n
                if split in cohorts: assert names==cohorts[split]
                else: cohorts[split]=names
    assert not cohorts['train'] & cohorts['validation']
    for r in cases:
        m=r['metrics']
        for k in ['inner','outer']:
            assert abs(m[k+'_both_correct']+m[k+'_both_wrong']+m[k+'_disagree']-1)<1e-12
        assert abs(m['cross_correct_transition']-m['cross_both_correct'])<1e-12
        assert m['bands_valid']==1
        threshold_float32=struct.unpack('f',struct.pack('f',.90))[0]
        assert bool(m['bands_confidence_adherent'])==(m['bands_truth']>=threshold_float32)
    keys=sorted(cases[0]['metrics'])
    aggregate={}
    for arm in ARMS:
        aggregate[arm]={}
        for k in keys:
            rows=[]
            for seed in range(3):
                t=[r['metrics'][k] for r in cases if r['arm']==arm and r['seed']==seed and r['split']=='train']
                v=[r['metrics'][k] for r in cases if r['arm']==arm and r['seed']==seed and r['split']=='validation']
                assert len(t)==10 and len(v)==52
                rows.append(dict(seed=seed,train=mean(t),validation=mean(v),gap=mean(t)-mean(v)))
            rows.append(dict(seed='Mean',**{f:mean([r[f] for r in rows]) for f in ['train','validation','gap']}))
            aggregate[arm][k]=rows
    compare={k:[dict(seed=seed,**{f:aggregate['separated'][k][i][f]-aggregate['pooled'][k][i][f] for f in ['train','validation','gap']}) for i,seed in enumerate([0,1,2,'Mean'])] for k in keys}
    drift={k:dict(mean_abs=mean([abs(r['bands_recomputed_drift'][k]) for r in cases]),max_abs=max(abs(r['bands_recomputed_drift'][k]) for r in cases)) for k in cases[0]['bands_recomputed_drift']}
    soft_drift={k:dict(mean_abs=mean(v),max_abs=max(v)) for k,v in soft_reproduction.items()}
    save(OUT/'NUMERICAL_REPRODUCTION.json',dict(hard_metrics_native_fraction_units=drift,soft_mse_native_units=soft_drift,interpretation='Small nonzero CUDA inference variation; CPU benchmark excluded. Original hard audit values retained for primary tables.'))
    save(OUT/'cases.json',cases)
    save(OUT/'temporal_cases.json',temporal_cases)
    save(OUT/'SUMMARY.json',dict(methodology=dict(fold=0,seeds=[0,1,2],train_cases=10,validation_cases=52,averaging='Case mean within seed, then arithmetic mean of three seeds. Gap always train minus validation.',stored_units='Correctness/agreement/Dice/fuzzy truth/adherence are fractions or [0,1] scores, not percentages. MSE/BCE are raw losses. Gaps use native metric units; multiply fraction gaps by100 for percentage points.',uncertainty='Descriptive; repeated seeds/cases/epochs are not independent samples; validation selected checkpoints, no independent test set.'),provenance=provenance,verified_original_files=len(manifest),bands_runtime=read(OUT/'bands_pooled_seed0.json')['runtime'],supplementary_historical_baseline='HISTORICAL_BASELINE.json',selected=aggregate,separated_minus_pooled=compare,recomputed_drift=drift,temporal=temporal))
    definitions=r'''# Selected-checkpoint constraint generalization audit

This is a descriptive analysis of the completed fold-0, three-seed experiment, using the original selected checkpoints. No training, checkpoint reselection, additional folds, or changes to prior result files were performed.

## Definitions and aggregation

Let y_i=1 for either annotated hippocampal class and 0 for background, p_i=P(anterior)+P(posterior), and h_i=1[argmax of the three class probabilities is anterior or posterior]. **The hard decoder is not a threshold of p_i at 0.5.** The inner band I is foreground removed by two six-connected erosions; the outer band O is background added by two six-connected dilations. Every positive-axis face adjacency is counted once, with both endpoints in I union O. E_in has both endpoints in I; E_out both in O; E_cross connects I and O.

- **Same-side hard equality satisfaction (agreement):** mean over E_in or E_out of 1[h_i=h_j]. This is 1-disagreement. Both endpoints can be wrong and still satisfy this relation.
- **Inner/outer pair correctness:** mean of 1[h_i=y_i and h_j=y_j] over the appropriate edge group. Inner requires two foreground predictions; outer requires two background predictions. This is stricter than agreement. Agreement = both-correct + both-wrong; correctness + both-wrong + disagreement = 1.
- **Crossing hard satisfaction/correctness:** mean over E_cross of 1[h_i-h_j=y_i-y_j]. This is exactly the saved `cross_correct_transition`, also equal to crossing both-correct. It is **not its complement**. Its complement is transition failure; an inverted transition is a subset of failures. This tests the exact annotated boundary face and orientation, so a roughly 25% rate does not mean only 25% of hippocampal voxels are correctly segmented; a small boundary displacement can fail many such faces while Dice remains high.
- **Soft edge error:** MSE_k = mean over E_k of ((p_i-p_j)-(y_i-y_j))^2. On same-side edges this is probability-difference MSE; on crossing edges it is signed-contrast MSE. Lower is better. No hard percentage is inferred from it. Hard squared error is another metric: a reversed crossing contributes 4, so it is not the complement of correct_transition.
- **Bands:** the frozen `OuterBoundaryBandLoss` uses B_in=mean_I[-log p_i], B_out=mean_O[-log(1-p_i)] and B=(B_in+B_out)/2 (implementation denominator count+1e-6). Focal and degree weights are zero in these runs. Its existing fuzzy truth is T=exp(-B), with side truths exp(-B_in) and exp(-B_out). Its existing **case-level** binary `confidence_adherent` is 1[T >= 0.90]. This is a predefined confidence threshold, not voxel classification or pair correctness. Reported adherence is the percentage of cases passing it. Fuzzy truth multiplied by 100 is a score, not a percent of correct edges. Mean case truth is computed before averaging, not exp(-mean B). B is unweighted BCE; the tiny weighted bands contribution in training logs is not this raw loss.
- **Dice:** hard macro Dice averages anterior and posterior class Dice equally per case. Binary union Dice is supplied separately for context; neither is the training soft Dice loss (which includes background).

Every selected-case metric is evaluated with the same definition on train and validation. Each seed first averages 10 train or 52 validation cases equally, then the three seed means are averaged equally. Edge/voxel counts do not weight patients. Gap always means train minus validation; positive gaps favor train for correctness/truth/Dice, but **negative gaps favor train for losses/disagreement**. Percentage metrics have gaps in percentage points (pp); fuzzy truths scaled by100 have gaps in score points. No repeated case, seed, or epoch is treated as an independent sample for a confidence claim. The same validation set selected checkpoints; this is not an independent test-set estimate.

## Selected checkpoints
'''
    report=[definitions,table(['Arm','Seed','Selected epoch','Stopped epoch'],[[r['arm'],r['seed'],r['selected_epoch'],r['stopped_epoch']] for r in provenance])]
    primary=[('Inner pair correctness','inner_both_correct'),('Inner equality satisfaction','inner_hard_agreement'),('Outer pair correctness','outer_both_correct'),('Outer equality satisfaction','outer_hard_agreement'),('Crossing correct_transition','cross_correct_transition'),('Bands fuzzy truth score','bands_truth'),('Bands case adherence >=0.90','bands_confidence_adherent'),('Dice (macro)','macro_dice')]
    for arm in ARMS:
        report.append(f'## {arm.capitalize()}: selected checkpoint values\n\nAll values below are percentages or fuzzy-truth score times100; gaps are percentage points or fuzzy score points, respectively. Bands adherence is case-level, not edge-level.')
        rows=[]
        for name,k in primary:
            for r in aggregate[arm][k]: rows.append([name,r['seed'],f"{r['train']*100:.6f}",f"{r['validation']*100:.6f}",f"{r['gap']*100:+.6f}"])
        report.append(table(['Constraint / metric','Seed','Train','Validation','Gap'],rows))
    report.append('## Soft losses and bands side metrics\n\nEach entry is **train / validation / gap**. These are raw losses or fuzzy truths; MSE and BCE are not hard satisfaction fractions. Lower loss is better.')
    for arm in ARMS:
        rows=[]
        for name,k in [('Inner edge MSE','inner_soft_squared_error'),('Outer edge MSE','outer_soft_squared_error'),('Crossing signed MSE','cross_soft_squared_error'),('Bands inner BCE','bands_inner_loss'),('Bands outer BCE','bands_outer_loss'),('Bands balanced BCE','bands_case_loss'),('Bands inner fuzzy truth','bands_inner_truth'),('Bands outer fuzzy truth','bands_outer_truth'),('Union Dice','union_dice')]:
            rows.append([name]+[' / '.join(f"{r[f]:.6f}" for f in ['train','validation','gap']) for r in aggregate[arm][k]])
        report.append(f'### {arm.capitalize()}\n\n'+table(['Metric','Seed 0','Seed 1','Seed 2','Mean'],rows))
    report.append('## Change when separating the rules\n\nSeparated minus pooled, averaging seeds. A smaller positive correctness gap is only beneficial if validation improves; reducing a gap by making training worse is not improved generalization.')
    rows=[]
    for name,k in primary:
        r=compare[k][-1]; rows.append([name]+[f"{r[f]*100:+.6f}" for f in ['train','validation','gap']])
    report.append(table(['Metric','Change train (points)','Change validation (points)','Change gap (points)'],rows))
    report.append('## Temporal evidence\n\nOnly validation has full-cohort hard audits at intermediate epochs. Intermediate model checkpoints were not retained, so full-training hard correctness cannot be reconstructed over time. The saved gradient audits provide exactly the same raw rule MSE definitions on **two fixed training cases**, in FP32 eval mode, at matching epochs; validation MSE uses the saved CUDA-AMP inference. This restricted comparison is informative about direction but is not a full-training trajectory or proof of when a full-cohort hard gap opens. Epoch-level training losses were measured during changing model updates; they are not substituted for same-checkpoint evaluations. All per-case temporal inputs and per-seed summaries are preserved. No trajectory p-values are computed.')
    for arm in ARMS:
        rows=[]
        for ep in [5,15,30,45,60]:
            q=[r for r in temporal if r['arm']==arm and r['epoch']==ep]; assert len(q)==3
            rows.append([ep]+[' / '.join(f'{v:.6f}' for v in [mean([r['train_probe'][k] for r in q]),mean([r['validation'][k+'_soft_squared_error'] for r in q]),mean([r['train_probe'][k]-r['validation'][k+'_soft_squared_error'] for r in q])]) for k in ['inner','outer','cross']]+[f"{mean([r['validation']['inner_both_correct'] for r in q])*100:.3f}",f"{mean([r['validation']['macro_dice'] for r in q])*100:.3f}"])
        report.append(f'### {arm.capitalize()} temporal means\n\nEach MSE entry is **two-training-case mean / validation mean / gap**.\n\n'+table(['Epoch','Inner MSE','Outer MSE','Crossing MSE','Val inner correct %','Val macro Dice %'],rows))
    report.append('## Provenance and numerical limits\n\nAll '+str(len(manifest))+' original result-manifest files were rehashed, and original source payload hashes were verified. Each selected audit matches its completion-manifest hash. All six downloaded checkpoint hashes and embedded run/epoch fields match their original completion manifests; dataset and split hashes match. Only retained selected weights were used. The original frozen dataset builder gives train and validation identical deterministic spatial preprocessing; augmentation is disabled. Recomputed bands metrics use the original frozen bands implementation and matching configuration, evaluated with the original CUDA-AMP model inference and FP32 bands computation for both splits. Original selected edge/Dice tables retain the original saved audit values. Recomputed full per-case edge/Dice metrics allow numerical-drift inspection. A local CPU-FP32/MONAI-version-mismatched benchmark showed substantial drift and was excluded; its diagnostic file is retained separately.\n\n'+table(['Recomputed versus saved metric','Mean absolute difference (pp)','Maximum case absolute difference (pp)'],[[k,f"{v['mean_abs']*100:.6f}",f"{v['max_abs']*100:.6f}"] for k,v in drift.items()]))
    report.append('The CUDA reproduction differences are small relative to the approximately 10–15-point correctness/confidence gaps, but they are not zero: no bitwise-reproducibility claim is made, and tiny arm differences remain descriptive. Recomputed soft edge-MSE mean absolute differences are approximately 2.08e-7 (inner), 3.41e-7 (outer), and 2.08e-6 (crossing); maximum case differences are 9.63e-6, 1.14e-5, and 4.77e-5 respectively. `NUMERICAL_REPRODUCTION.json` preserves these checks. There were no previously saved selected-checkpoint bands outputs against which to compute direct bands drift.')
    report.append(f'The new `cases.json` preserves every original selected-case metric plus the separately recomputed bands metrics and drift checks. `bands_*_seed*.json` retain all inference outputs; `temporal_cases.json` preserves intermediate case-level evidence; `SUMMARY.json` contains full precision per-seed and mean gaps, arm differences, temporal summaries and checkpoint provenance. Source definitions: [edge audit]({ROOT}/reports/separated_edge_20260929/source/scripts/audit_edge_coherence.py), [original bands]({ROOT}/reports/separated_edge_20260929/source/thesis/new_constraints/bands/outer_boundary.py), and [separated edge loss]({ROOT}/reports/separated_edge_20260929/source/thesis/new_constraints/separated_edge.py).')
    (OUT/'TABLES_AND_METHODS.md').write_text('\n\n'.join(report)+'\n')
    print(json.dumps(dict(drift=drift,means={a:{k:v[-1] for k,v in aggregate[a].items() if k in ['bands_truth','bands_inner_truth','bands_outer_truth','bands_case_loss','bands_confidence_adherent','inner_both_correct','inner_hard_agreement','outer_both_correct','outer_hard_agreement','cross_correct_transition','macro_dice']} for a in ARMS}),indent=2))

if __name__=='__main__': main()
