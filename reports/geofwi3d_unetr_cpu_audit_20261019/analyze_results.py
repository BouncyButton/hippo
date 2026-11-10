"""Recompute reported metrics from archived counts and analyze all UNETR curves."""
import csv
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np
from scipy.stats import spearmanr

HERE=Path(__file__).resolve().parent
BUNDLE=HERE.parent/'geofwi3d_colab_seed_search_20261018/bundle'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def describe(a):
    a=np.array(a,dtype=float)
    return dict(n=len(a),mean=float(a.mean()),sd=float(a.std(ddof=1)) if len(a)>1 else 0,
                median=float(np.median(a)),min=float(a.min()),max=float(a.max()))
manifest=read(BUNDLE/'data/fold_0_manifest.json');splits=read(BUNDLE/'splits.json')
features={r['case']:{k:float(v) for k,v in r.items() if k!='case'} for r in csv.DictReader((HERE/'data_case_features.csv').open())}
run_rows=[];case_rows=[];curves={};verified_files=0;source_matches=0
sources={'script_sha256':'scripts/cross_dataset_campaign/capped_core.py',
 'objective_source_sha256':'scripts/compute_constraint_objectives.py',
 'shared_source_sha256':'scripts/train_compute_matched_baselines.py',
 'combined_objective_sha256':'scripts/capped_depth_delay/objectives.py'}
for folder in sorted((HERE/'evidence').iterdir()):
    archive=read(folder/'archive_manifest.json');config=read(folder/'config.json');done=read(folder/'completion.json')
    t=archive['task'];model=t['model'];seed=t['seed'];fold=t['fold']
    for path in folder.iterdir():
        if path.name in archive['metadata']:
            assert sha(path)==archive['metadata'][path.name]['sha256'],str(path)
            verified_files+=1
    for key,path in sources.items():
        assert config[key]==sha(BUNDLE/'source'/path),(folder,key)
        source_matches+=1
    assert config['train_cases']==splits[fold]['train'] and config['validation_cases']==splits[fold]['val']
    assert config['model_identity']==model and config['seed']==seed and config['fold']==fold
    assert config['augmentation']=='none' and not config['smoke_test']
    assert config['torch']=='2.12.1+cu130' and config['monai']=='1.6.0'
    assert done['discarded_attempts']==done['calibration_attempts']==0
    assert done['updates']==config['schedule']['max_steps']
    assert done['consumed_training_flops']==done['updates']*config['schedule']['flops_per_update']
    assert done['unused_flops']<config['schedule']['flops_per_update']
    rows=list(csv.DictReader((folder/'metrics.csv').open()))
    for r in rows:
        for k,v in r.items():r[k]=float(v)
    curves[folder.name]=rows
    best=max(rows,key=lambda r:r['macro_hard_dice']);last=rows[-1]
    assert last['step']==done['updates'] and abs(best['macro_hard_dice']-done['best_validation_hard_dice'])<1e-12
    stats={}
    for filename,split_name in [('validation_latest.json','val'),('validation_best.json','best_val'),('training_final.json','train')]:
        doc=read(folder/filename);cases=doc['cases'];names=splits[fold]['train' if split_name=='train' else 'val']
        assert len(cases)==len(names) and {r['case_name'] for r in cases}==set(names)
        scores=[];precisions=[];recalls=[];aggregate=np.zeros((2,2),dtype=np.int64)
        for r in cases:
            m=np.array(r['confusion']);tn,fp,fn,tp=map(int,m.flatten());name=r['case_name']
            assert m.shape==(2,2) and (m>=0).all() and int(m.sum())==96**3
            assert fn+tp==manifest['cases'][name]['salt_voxels'],(folder,name)
            dice=2*tp/(2*tp+fp+fn);precision=tp/(tp+fp) if tp+fp else 0;recall=tp/(tp+fn)
            assert abs(dice-r['macro_hard_dice'])<1e-12
            scores.append(dice);precisions.append(precision);recalls.append(recall);aggregate+=m
            case_rows.append(dict(model=model,seed=seed,fold=fold,split=split_name,case=name,
                dice=dice,precision=precision,recall=recall,tn=tn,fp=fp,fn=fn,tp=tp,
                predicted_fraction=(tp+fp)/m.sum(),true_fraction=(tp+fn)/m.sum()))
        assert abs(np.mean(scores)-doc['summary']['macro_hard_dice'])<1e-12
        tn,fp,fn,tp=map(int,aggregate.flatten())
        stats[split_name]=dict(dice=float(np.mean(scores)),micro_dice=2*tp/(2*tp+fp+fn),
            precision=tp/(tp+fp),recall=tp/(tp+fn),mean_case_precision=float(np.mean(precisions)),
            mean_case_recall=float(np.mean(recalls)),predicted_fraction=(tp+fp)/aggregate.sum(),
            true_fraction=(tp+fn)/aggregate.sum(),volume_ratio=(tp+fp)/(tp+fn),
            fp_to_fn=fp/fn,missed_salt_fraction=fn/(tp+fn),false_discovery_fraction=fp/(tp+fp),
            cases_below_20=sum(x<.2 for x in scores),cases_above_50=sum(x>.5 for x in scores))
    assert abs(stats['val']['dice']-last['macro_hard_dice'])<1e-12
    assert abs(stats['train']['dice']-done['final_training_hard_dice'])<1e-12
    run_rows.append(dict(model=model,seed=seed,fold=fold,final_dice=stats['val']['dice'],
        best_dice=best['macro_hard_dice'],best_step=int(best['step']),best_budget_fraction=best['compute_fraction'],
        train_dice=stats['train']['dice'],peak_to_final_drop=best['macro_hard_dice']-stats['val']['dice'],
        last20pct_change=last['macro_hard_dice']-min(rows,key=lambda r:abs(r['compute_fraction']-.8))['macro_hard_dice'],
        loss_early=rows[0]['mean_train_loss'],loss_final=last['mean_train_loss'],
        final_lr=last['learning_rate'],updates=done['updates'],stats=stats))

unetr=sorted([r for r in run_rows if r['model']=='unetr_vit_b16'],key=lambda r:r['seed'])
summary={}
for model in sorted({r['model'] for r in run_rows}):
    rows=[r for r in run_rows if r['model']==model]
    summary[model]=dict(seeds=[r['seed'] for r in rows],
        curves={k:describe([r[k] for r in rows]) for k in ['final_dice','best_dice','train_dice','best_step','best_budget_fraction','peak_to_final_drop','last20pct_change']},
        errors={s:{k:describe([r['stats'][s][k] for r in rows]) for k in rows[0]['stats'][s]} for s in ['train','val','best_val']})
avg_curves=[]
for step in [int(r['step']) for r in curves['GeoFWI3D_unetr_vit_b16_baseline_f0s0']]:
    rs=[next(r for r in curves['GeoFWI3D_unetr_vit_b16_baseline_f0s'+str(t['seed'])] if int(r['step'])==step) for t in unetr]
    avg_curves.append(dict(step=step,budget_fraction=rs[0]['compute_fraction'],
        dice_mean=float(np.mean([r['macro_hard_dice'] for r in rs])),
        dice_min=min(r['macro_hard_dice'] for r in rs),dice_max=max(r['macro_hard_dice'] for r in rs),
        loss_mean=float(np.mean([r['mean_train_loss'] for r in rs])),learning_rate=rs[0]['learning_rate']))
ucases=[r for r in case_rows if r['model']=='unetr_vit_b16' and r['split']=='val']
case_avg=[]
for name in splits[0]['val']:
    rows=[r for r in ucases if r['case']==name]
    case_avg.append(dict(case=name,dice_mean=float(np.mean([r['dice'] for r in rows])),
        dice_min=min(r['dice'] for r in rows),dice_max=max(r['dice'] for r in rows),
        seeds_above_50=sum(r['dice']>.5 for r in rows),**features[name]))
correlations={k:dict(rho=float(spearmanr([r[k] for r in case_avg],[r['dice_mean'] for r in case_avg]).statistic))
              for k in ['salt_fraction','centroid_z','fault_present','body_count','salt_abs_mean','background_abs_mean','normalization_scale']}
matched={}
for model in ['nnunet','swinunetr']:
    comparisons=[]
    for seed in [0,14,18]:
        other={r['case']:r['dice'] for r in case_rows if r['model']==model and r['seed']==seed and r['split']=='val'}
        ours={r['case']:r['dice'] for r in ucases if r['seed']==seed}
        comparisons.append(dict(seed=seed,cases_other_better=sum(other[c]>ours[c] for c in ours),
             mean_gap=float(np.mean([other[c]-ours[c] for c in ours]))))
    matched[model]=comparisons
report=dict(status='CPU audit complete; causal training hypotheses remain untested',
    evidence_runs=len(run_rows),archive_file_hashes_verified=verified_files,source_hash_checks=source_matches,
    all_metric_counts_and_case_ground_truth_verified=True,models=summary,
    unetr_all_peaks_below_50=all(r['best_dice']<.5 for r in unetr),
    unetr_peaks_before_final=sum(r['best_step']<414 for r in unetr),
    unetr_declines_in_last20pct=sum(r['last20pct_change']<0 for r in unetr),
    unetr_training_loss_decreases=sum(r['loss_final']<r['loss_early'] for r in unetr),
    mean_curve=avg_curves,case_feature_correlations=correlations,matched_comparisons=matched,
    worst_cases=sorted(case_avg,key=lambda r:r['dice_mean'])[:10],
    always_below_50_cases=sum(r['dice_max']<=.5 for r in case_avg),
    never_above_20_cases=sum(r['dice_max']<.2 for r in case_avg),
    runs=run_rows)
(HERE/'results_audit.json').write_text(json.dumps(report,indent=2)+'\n')
for name,rows in [('case_metrics.csv',case_rows),('case_average_features.csv',case_avg),('mean_unetr_curve.csv',avg_curves)]:
    with (HERE/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps({k:v for k,v in report.items() if k not in ['runs','models','worst_cases','mean_curve']},indent=2))
for model,x in summary.items():
    print(model,'CURVES',json.dumps(x['curves']))
    print(model,'ERRORS',json.dumps({s:{k:v['mean'] for k,v in x['errors'][s].items()} for s in x['errors']}))
